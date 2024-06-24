import pickle
import torch
from dnnlib.util import open_url


class MC_Score_EDM(torch.nn.Module):

    def __init__(self, network_pkl, t: float, n_mc_samples: int = 10):
        super().__init__()
        self.t = t
        self.n_mc_samples = n_mc_samples
        with open_url(network_pkl, verbose=True) as f:
            self.net = pickle.load(f)['ema']


    def score_y_x(self, x: torch.Tensor, y: torch.Tensor, sigma_min=0.002, sigma_max=80,) -> torch.Tensor:
        
        assert self.net.label_dim, "net must be conditional. found: {}".format(self.net.label_dim)

        # make sure y is one-hot encoded, duplicate for parallel processing
        y = torch.eye(self.net.label_dim, device=y.device)[y].to(x) # assume that y is flat tensor of ints/longs
        y_dup = y[:, None, :].tile(1, self.n_mc_samples, 1).view(-1)
        # "duplicate" again for classifier-free guidance
        y_dup = torch.cat([y_dup, torch.zeros_like(y_dup)], dim=0)

        # Adjust noise levels based on what's supported by the network.
        sigma_min = max(sigma_min, self.net.sigma_min)
        sigma_max = min(sigma_max, self.net.sigma_max)
        sigma = (sigma_max - sigma_min) * self.t + sigma_min
        sigma = self.net.round_sigma(sigma).to(x) # following from generate.ema_sampler

        # duplicate x by n_mc_samples
        x_dup = x[:, None, :, :, :].tile(1, self.n_mc_samples, 1, 1, 1)
        x_dup_shp = x_dup.shape
        # add noise
        x_dup = x_dup + torch.randn_like(x_dup) * sigma
        # flatten x_dup for batch processing
        x_dup = x_dup.view((-1,) + x_dup_shp[-3:])
        # duplicate again by 2 along batch dimension for classifier-free guidance
        x_dup = torch.cat([x_dup, x_dup], dim=0)

        # evaluate the diffusion model with proper conditioning. remember: DENOISER FUNCTION in this impl
        D_x_dup = self.net(x_dup, sigma, y_dup)
        scores_dup = (D_x_dup - x_dup) / (sigma ** 2)
        scores_dup_cond, scores_dup_uncond = scores_dup.chunk(2)

        scores_dup = (scores_dup_cond - scores_dup_uncond).view(x_dup_shp)

        return scores_dup.mean(dim=1)


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.score_y_x(x, y)