
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import numpy as np


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('filename', type=str)
    parser.add_argument('--n_disp', type=int, default=64)


    args = parser.parse_args()

    ims = np.load(args.filename)
    if 'image' in ims.files:
        ims = ims['image']

    sq = int(float(args.n_disp)**0.5 + 0.5)

    fig, axs = plt.subplots(sq, sq)

    for i in range(sq):
        for j in range(sq):
            if i*sq + j < args.n_disp:
                axs[i][j].imshow(ims[i*sq + j])

            axs[i][j].xaxis.set_visible(False)
            axs[i][j].yaxis.set_visible(False)

    fig.tight_layout()

    fig.savefig('./figs/vis_samp.png')

