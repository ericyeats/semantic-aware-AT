import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from core.data import load_data
from core.data import SEMISUP_DATASETS

from argparse import ArgumentParser
from core.utils.utils import str2bool, str2float
from core.data import DATASETS

from mc_score import MC_Score_EDM


def main(args):

    SCORE_DATA_NAME = args.score_data_name
    if args.score_data_name is None:
        SCORE_DATA_NAME = "{}_score_t{:1.2f}_mc{}".format(args.data, args.time, args.n_mc_samples)

    SCORE_DIR = os.path.join(args.data_dir, SCORE_DATA_NAME)

    if not os.path.exists(SCORE_DIR):
        os.makedirs(SCORE_DIR)

    DATA_DIR = os.path.join(args.data_dir, args.data)
    BATCH_SIZE = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load in the training data. do not shuffle it
    val = args.data in SEMISUP_DATASETS
    loaded_data = load_data(
        DATA_DIR, BATCH_SIZE, BATCH_SIZE, use_augmentation='none', use_consistency=args.consistency, shuffle_train=False, 
        aux_data_filename=args.aux_data_filename, unsup_fraction=args.unsup_fraction, validation=val
    )
    if val:
        train_dataset, test_dataset, eval_dataset, train_dataloader, test_dataloader, eval_dataloader = loaded_data
        del train_dataset, test_dataset, eval_dataset, test_dataloader, eval_dataloader
    else:
        train_dataset, test_dataset, train_dataloader, test_dataloader = loaded_data
        del train_dataset, test_dataset, test_dataloader


    score_net = MC_Score_EDM(args.score_network_pkl, args.time, args.n_mc_samples)
    score_net = nn.DataParallel(score_net).to(device)

    # begin loading data

    for i, batch in enumerate(train_dataloader):

        x, y = batch
        x = x.to(device)
        y = y.to(device)

        x = x * 2. - 1. # normalize appropriately
        with torch.no_grad():
            x_score = score_net(x, y)

        # un-normalize and save. for dlogp(x)/dx, multiply by 2.
        # save as a numpy array to conserve memory
        x_score = (x_score * 2.).detach().cpu().numpy()

        # save each precomputed batch as a np file. these will be loaded and aggregated into a dataset later
        np.save(os.path.join(SCORE_DIR, 'batch_{}_scores.npy'.format(i)), x_score)

        torch.cuda.empty_cache()

    print("Done computing scores...")

    # load all scores and images and combine them
    image_score = []
    for i, (x, y) in enumerate(train_dataloader):
        x_score = np.load(os.path.join(SCORE_DIR, 'batch_{}_scores.npy'.format(i)))
        x_score = torch.from_numpy(x_score)

        # concat original image and score along the channel dimension
        x_combined = torch.cat([x, x_score], dim=1)
        image_score.append(x_combined)

    # combine all batches into a single tensor
    image_score = torch.cat(image_score, dim=0)
    # save to disk
    save_path = os.path.join(SCORE_DIR, 'cifar10_with_scores.pt')
    print(f"Attempting to save file to: {save_path}")
    torch.save(image_score, save_path)

    if os.path.exists(save_path):
        print(f"File successfully saved at: {save_path}")
        print(f"File size: {os.path.getsize(save_path)} bytes")
    else:
        print(f"Failed to save file at: {save_path}")

    print("Done saving combined data...")


if __name__ == "__main__":

    parser = ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='~/data')
    parser.add_argument('-d', '--data', type=str, default='cifar10', choices=DATASETS, help='Data to use.')
    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input.')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for evaluation.')
    parser.add_argument('--score_data_name', type=str, default=None, help='Name of folder to save the score data in. Defaults to <data>_score_t<time>_mc<n_mc_samples>')

    parser.add_argument('--score_network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str)
    parser.add_argument('--time', help='Time in [0, 1] to which data should be diffused for average scores. default 0.1', type=float, default=0.1)
    parser.add_argument('--n_mc_samples', type=int, default=40, help='Number of samples for average score calculation')
    parser.add_argument('--n_chunks', type=int, default=10, help='Number of chunks for the score estimation network forward pass')

    parser.add_argument('--consistency', type=str2bool, default=False, help='Use consistency in data loading')
    parser.add_argument('--aux_data_filename', type=str, default=None, help='Filename for auxiliary data')
    parser.add_argument('--unsup_fraction', type=float, default=None, help='Fraction of unsupervised data to use')

    args = parser.parse_args()

    main(args)

    exit(0)


