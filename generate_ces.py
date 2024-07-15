"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

htanh = torch.nn.Hardtanh()

def boltz_score(x: torch.Tensor, sigma_g: torch.Tensor, mu_c: torch.Tensor, sigma_c: torch.Tensor) -> torch.Tensor:
    """
    x: input tensor of (B, C, H, W)
    sigma_g: gaussian noise scale tensor of (B,), positive
    mu_c: adaptive distribution loc tensor of (B, C, H, W)
    sigma_c: adaptive distribution scale tensor of (B,)
    """
    y = x - mu_c
    u = sigma_g.type(torch.double)/sigma_c.type(torch.double)
    cutoff_s_val = 35.493278272052976
    cutoff_u_val = 20.
    f = torch.where(
        u >= cutoff_u_val,
        cutoff_s_val + (u-cutoff_u_val)*(np.pi**0.5),
        (torch.exp(-(u**2))/(torch.erfc(u)))
    )
    slope = ((2**0.5)/(sigma_c) - (2**0.5)/(sigma_g*(np.pi**0.5))*f).type(torch.float)
    height = (2**0.5)/sigma_c
    return height[:, None, None, None]*htanh(slope[:, None, None, None]*y)

BASE_DATASETS = ["cifar10"] # more later

def percentile_clamp(x, bnd=1., perc=0.9):

    if perc < 1.0:
        assert perc > 0.
        # sort the pixels in the batch
        x_sorted, indices = torch.sort(x.view(x.shape[0], -1).abs(), dim=-1)
        elem_perc = x_sorted[:, int(x_sorted.shape[1] * perc)][:, None, None, None]
        
        # determine if threshs are greater than bnd
        thresh_exceed = elem_perc > bnd
        # thresh > bnd, clamp to threshold, rescale to bnd
        x = torch.where(
            thresh_exceed.expand_as(x),
            torch.clamp(x, -elem_perc.expand_as(x), elem_perc.expand_as(x)) / elem_perc * bnd,
            x
        )
    return x
#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_ce_sampler(
    net, net_uncond, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    mu_ce=None, ce_sigma=0.2, guidance=0., perc=1.0):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    B = latents.shape[0]
    dev = latents.device

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        if guidance > 0.:
            denoised_uncond = net_uncond(x_hat, t_hat, None).to(torch.float64)
            d_cur_uncond = (x_hat - denoised_uncond) / t_hat
            d_cur += guidance * (d_cur - d_cur_uncond) # classifier-free guidance

        if mu_ce is not None:
            var_t = t_hat**2
            var_max = sigma_max**2
            d_cur += t_hat * -boltz_score(x_hat, t_hat*torch.ones((B,), device=dev), mu_ce, ce_sigma*torch.ones((B,), device=dev))
            # d_cur += t_hat * (x_hat - mu_ce) / (ce_sigma**2 + var_t) # d_cur approx sigma_t * -score
        
        # calculate x_orig and apply dynamic thresholding
        x_orig = x_hat - t_hat * d_cur
        x_orig = percentile_clamp(x_orig, 1., perc)
        x_next = x_orig + (t_next / t_hat) * (x_hat - x_orig)
        # x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next

            if guidance > 0.:
                denoised_uncond = net_uncond(x_next, t_next, None).to(torch.float64)
                d_prime_uncond = (x_next - denoised_uncond) / t_next
                d_prime += guidance * (d_prime - d_prime_uncond) # classifier-free guidance

            if mu_ce is not None:
                var_t = t_next**2
                var_max = sigma_max**2
                # d_prime += t_next * (x_next - mu_ce) / (ce_sigma**2 + var_t) # d_cur approx sigma_t * -score
                d_prime += t_next * -boltz_score(x_next, t_next*torch.ones((B,), device=dev), mu_ce, ce_sigma*torch.ones((B,), device=dev))

            x_orig = x_hat - t_hat * (0.5 * d_cur + 0.5 * d_prime)
            x_orig = percentile_clamp(x_orig, 1., perc)
            x_next = x_orig + (t_next / t_hat) * (x_hat - x_orig)
            # x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next



#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--network_uncond', 'network_pkl_uncond',  help='Network pickle filename', metavar='PATH|URL',        type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--class_num', 'class_num', help='Number of the class', metavar='INT',                                type=click.IntRange(min=1), default=10, show_default=True)

# base dataset indices (replaced seeds)
@click.option('--base_indices',            help='Indices (e.g. 1,2,5-10) to grab data from dataset', metavar='LIST',type=parse_int_list, default='0-63', show_default=True)

# sampler args
@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--ce_sigma',                help='Scale of the CE neighborhood distribution', metavar='FLOAT',       type=float, default=0.2, show_default=True)
@click.option('--guidance',       help='Classifier-free guidance', metavar='FLOAT',                        type=float, default=10., show_default=True)
@click.option('--perc',                    help='Guidance clipping percentile',                                     type=float, default=1.0, show_default=True)

# other CE-related args
@click.option('--base_dataset_root',       help='Path to base datasets folder', metavar='PATH',                     type=str, default='~/data')
@click.option('--base_dataset',            help='Base dataset for CE generation', metavar='STR',                    type=str, default='cifar10')
@click.option('--n_ces',                   help='Number of CEs per class', metavar='INT',                           type=int, default=2, show_default=True)

@click.option('--verbose', is_flag=True)


def main(network_pkl, network_pkl_uncond, outdir, subdirs, base_indices, class_num, max_batch_size, base_dataset_root, base_dataset, n_ces, verbose, device=torch.device('cuda'), **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate_ces.py --outdir=out --base_indices=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate_ces.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """

    dist.init()

    num_batches = ((len(base_indices) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size() # total number of batches across all devices
    all_batches = torch.as_tensor(base_indices).tensor_split(num_batches) # splits the seeds list into batches of seeds
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()] # get the seed batches corresponding to this device

    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)
    # Load unconditional network.
    dist.print0(f'Loading unconditional network from "{network_pkl_uncond}"...')
    with dnnlib.util.open_url(network_pkl_uncond, verbose=(dist.get_rank() == 0)) as f:
        net_uncond = pickle.load(f)['ema'].to(device)

    # load the training dataset for the CEs
    assert base_dataset in BASE_DATASETS

    if base_dataset == 'cifar10':
        base_dataset = CIFAR10(os.path.join(base_dataset_root, base_dataset), train=True, transform=ToTensor(), download=False)
    else:
        raise ValueError(f"{base_dataset} not recognized")

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

        # make dir
        os.makedirs(outdir, exist_ok=True)

    # Loop over batches.
    dist.print0(f'Generating {len(base_indices) * n_ces} images to "{outdir}"...')

    for class_idx in range(class_num):
        for ce_idx in range(n_ces):
            dist.print0(f'Generating Class {class_idx} CE {ce_idx}...')

            # final output.
            total_image_np = []

            # iterate through the base_dataset
            for batch_indices in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0 or not verbose)):
                torch.distributed.barrier()
                batch_size = len(batch_indices)
                if batch_size == 0:
                    continue

                # get the data for this rank. use this to adjust the sampling function for ce_generation
                base_data = []
                for b_i in batch_indices:
                    samp, _ = base_dataset[b_i]
                    base_data.append(samp[None, :, :, :])
                base_data = torch.cat(base_data, dim=0).to(device)
                # normalize the base_data
                mu_ce = 2.*base_data - 1.

                # Pick latents and labels.
                rnd = StackedRandomGenerator(device, batch_indices + (ce_idx*len(base_indices)))
                latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
                class_labels = None
                if net.label_dim:
                    class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
                if class_idx is not None:
                    class_labels[:, :] = 0
                    class_labels[:, class_idx] = 1

                # Generate images.
                sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}

                sampler_fn = edm_ce_sampler
                images = sampler_fn(net, net_uncond, latents, class_labels, randn_like=rnd.randn_like, mu_ce=mu_ce, **sampler_kwargs)

                # Save images.
                images_np = (images * 127.5 + 128).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()

                images_np_list = [torch.zeros_like(images_np) for _ in range(dist.get_world_size())]
                torch.distributed.all_gather(images_np_list, images_np)
                total_image_np.append(np.concatenate([i.cpu().numpy() for i in images_np_list]))

            # Done.
            torch.distributed.barrier()
            if dist.get_rank() == 0:
                np.save(os.path.join(outdir, f'{class_idx}_{ce_idx}.npy'), np.concatenate(total_image_np))
    
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
