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
from torch.autograd.functional import vjp
from typing import Callable

BASE_DATASETS = ["cifar10"] # more later

def get_ce_sampler(
    net, latents, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    ce_sigma=0.2) -> Callable[[torch.Tensor], torch.Tensor]:
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # make sure all the inputs do not require grad (except mu_ce)
    latents.requires_grad_(False)


    class MuCEDifferentiableSamplingFn(torch.autograd.Function):

        @staticmethod
        def forward(ctx, mu_ce):
            saved_tens = []
            # Main sampling loop.
            x_next = latents.to(torch.float64) * t_steps[0] # x_T
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1

                x_cur = x_next
                if mu_ce.requires_grad:
                    saved_tens.append(x_cur.detach().cpu()) # for gradient computation later
                
                var_t = t_cur**2

                # Euler step.
                with torch.no_grad():
                    denoised = net(x_cur, t_cur, None).to(torch.float64)
                dx_dt = (denoised - x_cur) / t_cur

                if mu_ce is not None:
                    dx_dt += t_cur * (mu_ce - x_cur) / (ce_sigma**2 + var_t) # d_cur approx sigma_t * -score

                x_next = x_cur + 2. * dx_dt * (t_cur - t_next)

                if i < num_steps - 2:
                    # add noise if not last step of denoising sequence
                    x_next = x_next + (2. * t_cur * (t_cur- t_next)) ** 0.5 * randn_like(x_cur)

                # apply eq (6) from Karras et al 2022 with beta_t = sigma_t_prime / sigma_t

                ctx.save_for_backward(*((mu_ce,) + tuple(saved_tens)))

            return x_next
        
        @staticmethod
        def backward(ctx, grad_out):
            back_state = ctx.saved_tensors
            mu_ce, xt_state = back_state[0], back_state[1:]

            agg_grad = torch.zeros_like(grad_out)

            for i, xt in enumerate(reversed(xt_state)):
                xt = xt.cuda()

                rev_ind = num_steps - 1 - i
                sigma_t = t_steps[rev_ind - 1]
                sigma_t_1 = t_steps[rev_ind]
                t_scl = 2. * (sigma_t - sigma_t_1)
                sigma_scl = sigma_t / (ce_sigma**2 + sigma_t**2)

                # compute the A Jacobian-vector product, aggregate
                A = lambda _mu_ce: t_scl * sigma_scl * _mu_ce
                grad_A = vjp(A, mu_ce, grad_out)[1]
                agg_grad += grad_A

                # compute the B Jacobian-vector product, pass back
                if i < num_steps - 1: # don't need to do this for x_T
                    B = lambda _xt: _xt + t_scl * (net(_xt, sigma_t, None).to(torch.float64) - _xt) / sigma_t - sigma_scl * t_scl * _xt
                    grad_B = vjp(B, xt, grad_out)[1]
                    grad_out = grad_B

            return agg_grad
    
    return MuCEDifferentiableSamplingFn.apply




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
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

# base dataset indices (replaced seeds)
@click.option('--base_indices',            help='Indices (e.g. 1,2,5-10) to grab data from dataset', metavar='LIST',type=parse_int_list, default='0-63', show_default=True)

# sampler args
@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)


@click.option('--ce_sigma',                help='Scale of the CE neighborhood distribution', metavar='FLOAT',       type=float, default=0.2, show_default=True)

# other CE-related args
@click.option('--base_dataset_root',       help='Path to base datasets folder', metavar='PATH',                     type=str, default='~/data')
@click.option('--base_dataset',            help='Base dataset for CE generation', metavar='STR',                    type=str, default='cifar10')
@click.option('--n_ces',                   help='Number of CEs per class', metavar='INT',                           type=int, default=2, show_default=True)
@click.option('--test', is_flag=True)

@click.option('--verbose', is_flag=True)


def main(network_pkl, outdir, subdirs, base_indices, max_batch_size, base_dataset_root, base_dataset, n_ces, test, verbose, device=torch.device('cuda'), **sampler_kwargs):
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

    # load the training dataset for the CEs
    assert base_dataset in BASE_DATASETS

    if base_dataset == 'cifar10':
        base_dataset = CIFAR10(os.path.join(base_dataset_root, base_dataset), train=not test, transform=ToTensor(), download=False)
    else:
        raise ValueError(f"{base_dataset} not recognized")

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

        # make dir
        os.makedirs(outdir, exist_ok=True)

    # Loop over batches.
    dist.print0(f'Generating {len(base_indices) * n_ces} images to "{outdir}"...')

    for ce_idx in range(n_ces):
        dist.print0(f'Generating CE {ce_idx}...')

        # final output.
        total_image_np = []
        total_label_np = []

        # iterate through the base_dataset
        for batch_indices in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0 or not verbose)):
            torch.distributed.barrier()
            batch_size = len(batch_indices)
            if batch_size == 0:
                continue

            # get the data for this rank. use this to adjust the sampling function for ce_generation
            base_data = []
            base_label = []
            for b_i in batch_indices:
                samp, lab = base_dataset[b_i]
                base_data.append(samp[None, :, :, :])
                base_label.append(lab)
            base_data = torch.cat(base_data, dim=0).to(device)
            base_label = torch.tensor(base_label, dtype=torch.long).to(device)
            assert len(base_label.shape) == 1
            base_data.requires_grad_(True)

            # normalize the base_data
            mu_ce = 2.*base_data - 1.

            # Pick latents and labels.
            rnd = StackedRandomGenerator(device, batch_indices + (ce_idx*len(base_indices)))
            latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)

            # Generate images.
            sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}

            sampler_fn = get_ce_sampler(net, latents, randn_like=rnd.randn_like, **sampler_kwargs)
            images = (sampler_fn(mu_ce + torch.randn_like(mu_ce) * 0.2) + 1) / 2.
            

            # # testing grad functionality
            # print("IMAGES GENERATED. Now Testing Grad")
            # base_data_grad = torch.autograd.grad(images.sum(), base_data)[0]
            # print(base_data_grad)

            # Save images.
            images_np = (images * 256.).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()

            images_np_list = [torch.zeros_like(images_np) for _ in range(dist.get_world_size())]
            labels_np_list = [torch.zeros_like(base_label) for _ in range(dist.get_world_size())]

            torch.distributed.all_gather(images_np_list, images_np)
            torch.distributed.all_gather(labels_np_list, base_label)
            total_image_np.append(np.concatenate([i.cpu().numpy() for i in images_np_list]))
            total_label_np.append(np.concatenate([l.cpu().numpy() for l in labels_np_list]))


        # Done.
        torch.distributed.barrier()
        if dist.get_rank() == 0:
            np.savez(os.path.join(outdir, f'{ce_idx}.npz'), image=np.concatenate(total_image_np), label=np.concatenate(total_label_np))
    
    dist.print0('Done.')

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------


# #--------------------------------------------------

# def continuous_sde_sampler(
#     net, latents, randn_like=torch.randn_like,
#     num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
#     mu_ce=None, ce_sigma=0.2):

#     # set up the solution using sdeint
#     shp = latents.shape
#     shp_flat = (shp[0], shp[1]*shp[2]*shp[3])

#     def tcv(t):
#         assert t >= 0. and t <= 1.
#         return sigma_max - (sigma_max - sigma_min) * t

#     class CE_SDE(torch.nn.Module):

#         sde_type = 'ito'
#         noise_type = 'diagonal'

#         def __init__(self, net, mu_ce, ce_sigma):
#             super().__init__()
#             self.mu_ce = mu_ce
#             self.ce_sigma = ce_sigma
#             self.net = net

#         def f(self, t, yt):
#             yt = yt.view(shp)
#             s = tcv(t)
#             # drift coefficient
#             sigma_data_score = (self.net(yt, s, None).double() - yt) / s
#             sigma_gauss_score = s * (self.mu_ce - yt) / (self.ce_sigma**2 + s**2)
#             return 2. * (sigma_data_score + sigma_gauss_score).view(shp_flat)

#         def g(self, t, yt):
#             s = tcv(t)
#             # diffusion coefficient
#             return torch.ones(shp_flat, dtype=torch.double, device='cuda') * (2. * s) ** 0.5

#     sde = CE_SDE(net, mu_ce, ce_sigma)

#     sol = torchsde.sdeint(sde, (latents.view(shp_flat)*sigma_max).double(), [0., 1.], dt=1e-2, method='euler')[1].view(shp)

#     return sol

