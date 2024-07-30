import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Tuple
import torch

def density_plots(arrs: List[np.ndarray], sig: float = 0.1, n_t: int = 1000) -> Tuple[np.ndarray, List[np.ndarray]]:
    t_min = min([np.min(a) for a in arrs])
    t_max = max([np.max(a) for a in arrs])

    t_min -= sig*5
    t_max += sig*5

    gauss_density = lambda x, mu, sig: ((sig**2 * 2. * np.pi)**-0.5)*np.exp(-0.5 * ((x - mu)/sig)**2)

    t = np.linspace(t_min, t_max, n_t)

    out = []
    for a in arrs:
        a_dense = np.zeros_like(t)
        for i, t_val in enumerate(t):
            a_dense[i] = np.mean(gauss_density(t_val, a, sig)) # uniform weighting
        out.append(a_dense)
    return t, out


if __name__ == "__main__":

    parser = ArgumentParser()

    # parser.add_argument("--time", type=float, default=0.1)
    parser.add_argument("--n_mc_samples", type=int, default=40)
    parser.add_argument("--sigma", type=float, default=0.1)

    args = parser.parse_args()

    times = [0.02, 0.05, 0.1, 0.2]

    score_norms_np = []

    for time in times:
        filename = "./score_data/cifar10score/score_t{:1.2f}_mc{}/data_with_scores.pt".format(time, args.n_mc_samples)
        _, score_data = torch.load(filename).chunk(2, dim=1)
        score_norms_np.append(score_data.view(score_data.shape[0], -1).norm(p=2, dim=1).cpu().numpy())


    t, dens = density_plots(score_norms_np, sig=args.sigma)


    plt.figure()

    for i, den in enumerate(dens):

        plt.plot(t, den, linewidth=2, label=times[i])

    plt.grid()
    plt.xlabel("Score Norm")
    plt.ylabel("Density")
    plt.legend()
    plt.xlim((-0.1, 2.1))

    plt.savefig('./figs/score_norm_densities.png')


