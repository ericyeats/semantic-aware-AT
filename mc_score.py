import pickle
import torch
from dnnlib.util import open_url


class MC_Score_EDM(torch.nn.Module):

    def __init__(self, network_pkl_cond, network_pkl_uncond, t: float, n_mc_samples: int = 10, n_chunks: int = 1):
        super().__init__()
        self.t = t
        self.n_mc_samples = n_mc_samples
        self.n_chunks = n_chunks
        assert self.n_chunks > 0
        with open_url(network_pkl_cond, verbose=True) as f:
            self.net_cond = pickle.load(f)['ema']
        with open_url(network_pkl_uncond, verbose=True) as f:
            self.net_uncond = pickle.load(f)['ema']

        


    def score_y_x(self, x: torch.Tensor, y: torch.Tensor, n_samples: int = 10, sigma_min=0.002, sigma_max=80,) -> torch.Tensor:
        
        assert self.net_cond.label_dim, "net must be conditional. found: {}".format(self.net_cond.label_dim)

        # make sure y is one-hot encoded, duplicate for parallel processing
        y = torch.eye(self.net_cond.label_dim, device=y.device)[y].to(x) # assume that y is flat tensor of ints/longs
        y_dup = y[:, None, :].tile(1, n_samples, 1).view(-1)
        # "duplicate" again for classifier-free guidance
        # y_dup = torch.cat([y_dup, torch.zeros_like(y_dup)], dim=0)

        # Adjust noise levels based on what's supported by the network.
        sigma_min_cond = max(sigma_min, self.net_cond.sigma_min)
        sigma_max_cond = min(sigma_max, self.net_cond.sigma_max)
        sigma_cond = (sigma_max_cond - sigma_min_cond) * self.t + sigma_min_cond
        sigma_cond = self.net_cond.round_sigma(sigma_cond).to(x) # following from generate.ema_sampler

        # Adjust noise levels based on what's supported by the network.
        sigma_min_uncond = max(sigma_min, self.net_uncond.sigma_min)
        sigma_max_uncond = min(sigma_max, self.net_uncond.sigma_max)
        sigma_uncond = (sigma_max_uncond - sigma_min_uncond) * self.t + sigma_min_uncond
        sigma_uncond = self.net_uncond.round_sigma(sigma_uncond).to(x) # following from generate.ema_sampler

        if sigma_cond != sigma_uncond:
            print("WARN - Unet variance schedules are different !")

        # duplicate x by n_samples
        x_dup = x[:, None, :, :, :].tile(1, n_samples, 1, 1, 1)
        x_dup_shp = x_dup.shape
        # add noise
        x_dup = x_dup + torch.randn_like(x_dup) * sigma_cond
        # flatten x_dup for batch processing
        x_dup = x_dup.view((-1,) + x_dup_shp[-3:])
        # duplicate again by 2 along batch dimension for classifier-free guidance
        # x_dup = torch.cat([x_dup, x_dup], dim=0)

        # evaluate the diffusion model with proper conditioning. remember: DENOISER FUNCTION in this impl
        D_x_dup_cond = self.net_cond(x_dup, sigma_cond, y_dup)
        scores_dup_cond = (D_x_dup_cond - x_dup) / (sigma_cond ** 2)

        D_x_dup_uncond = self.net_uncond(x_dup, sigma_uncond, None)
        scores_dup_uncond = (D_x_dup_uncond - x_dup) / (sigma_uncond ** 2)
        # scores_dup_cond, scores_dup_uncond = scores_dup.chunk(2)

        scores_dup = (scores_dup_cond - scores_dup_uncond).view(x_dup_shp)

        return scores_dup.mean(dim=1)


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # estimate scores in chunks
        results = torch.zeros_like(x)

        # distribute the job evenly over the chunks
        base_chunk_size = self.n_mc_samples//self.n_chunks # rounds down
        remainder = self.n_mc_samples - (base_chunk_size*self.n_chunks)
        for i in range(self.n_chunks):
            cur_chunk_size = base_chunk_size + (1 if i < remainder else 0)
            results += cur_chunk_size*self.score_y_x(x, y, n_samples=cur_chunk_size) 
                
        results /= self.n_mc_samples

        return results