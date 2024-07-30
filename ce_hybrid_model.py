import torch
import dnnlib
import pickle
from torch.nn import Module
from math import log as math_log
from utils_semisup import get_model
from typing import Callable
from torch.autograd.functional import vjp

def load_edm_from_path(edm_path):
    # Load network.
    print(f'Loading network from "{edm_path}"...')
    with dnnlib.util.open_url(edm_path, verbose=True) as f:
        net = pickle.load(f)['ema'].cuda()
    return net

def load_cls_from_path(cls_path, model_info='wrn-28-10', class_num=10):
    # Loading model
    checkpoint = torch.load(cls_path)
    num_classes = checkpoint.get('num_classes', class_num)
    normalize_input = checkpoint.get('normalize_input', False)
    model = get_model(model_info, 
                    num_classes=num_classes,
                    normalize_input=normalize_input)
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint['state_dict']) # not being DataParallel might break this
    model = model.module # discard the DataParallel aspect
    model.eval()
    return model.cuda()


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



class CEClassifier(Module):

    def __init__(self, edm, cls, n_ces=1, **sampling_kwargs):
        super(CEClassifier, self).__init__()
        self.edm = edm
        self.cls = cls
        self.n_ces = n_ces
        self.sampling_kwargs = sampling_kwargs

    def sample_ces(self, x, latents=None, randn_like=torch.randn_like):

        # duplicate x by n_ces times, stack along batch dimension for chunk later
        x = x.tile(self.n_ces, 1, 1, 1)

        # initialize latents with some random noise.
        if latents is None or (isinstance(latents, torch.Tensor) and latents.shape[0] != x.shape[0]):
            latents = randn_like(x)

        # normalize x for edm
        x = 2. * x - 1.

        ce_sampler_fn = get_ce_sampler(self.edm, latents, randn_like, **self.sampling_kwargs)

        x_ces = ce_sampler_fn(x).float()
        
        # x_ces = torch.clamp((x_ces + 1.) / 2., 0., 1.) # un-normalize the ces, clamp to image bounds.
        x_ces = (x_ces + 1.) / 2. # skip the clamp to avoid any potential gradient obfuscation

        # measure CE_Energy(x)
        ce_dist = torch.distributions.Normal(loc=x, scale=self.sampling_kwargs["ce_sigma"]/2.)
        ce_lps = ce_dist.log_prob(x_ces).sum(dim=(1,2,3))
        

        return x_ces, ce_lps

    def forward(self, x, latents=None, randn_like=torch.randn_like, ret_energy=False):   
        
        x_ces, ce_lps = self.sample_ces(x, latents, randn_like)

        # provide the CEs to the classification model for p(y|x)
        logits = self.cls(x_ces)
        logits = torch.logsumexp(logits.view((self.n_ces, x.shape[0], logits.shape[1])), dim=0) - math_log(self.n_ces)

        if ret_energy:
            ce_lps = torch.logsumexp(ce_lps.view((self.n_ces, x.shape[0])), dim=0) - math_log(self.n_ces)
            return logits, ce_lps
        
        return logits

