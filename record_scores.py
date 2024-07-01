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
        SCORE_DATA_NAME = "{}_score_t{:1.2f}_mc{}".format(args.data, args.score_t, args.n_mc_samples)

    SCORE_DIR = os.path.join(args.data_dir, SCORE_DATA_NAME)

    if not os.path.exists(SCORE_DIR):
        os.makedirs(SCORE_DIR)

    DATA_DIR = os.path.join(args.data_dir, args.data)
    BATCH_SIZE = args.batch_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        x_score = score_net(x, y)

        # un-normalize and save. for dlogp(x)/dx, multiply by 2.
        x_score = (x_score * 2.).cpu()

        torch.save(x_score, os.path.join(SCORE_DIR, 'batch_{}.pt'.format(i)))
    


    


if __name__ == "main":

    parser = ArgumentParser()

    parser.add_argument('--data-dir', type=str, default='~/data')
    parser.add_argument('-d', '--data', type=str, default='cifar10', choices=DATASETS, help='Data to use.')
    parser.add_argument('--normalize', type=str2bool, default=False, help='Normalize input.')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size for evaluation.')
    parser.add_argument('--score_data_name', type=str, default=None, help='Name of folder to save the score data in. Defaults to <data>_score_t<time>_mc<n_mc_samples>')

    parser.add_argument('--score_network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str)
    parser.add_argument('--time', help='Time in [0, 1] to which data should be diffused for average scores. default 0.1', type=float, default=0.1)
    parser.add_argument('--n_mc_samples', type=int, default=20, help='Number of samples for average score calculation')
    parser.add_argument('--n_chunks', type=int, default=4, help='Number of chunks for the score estimation network forward pass')


    args = parser.parse_args()

    main(args)

    exit(0)


