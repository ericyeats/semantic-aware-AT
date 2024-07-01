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

        x_score = score_net(x, y)

        # un-normalize and save. for dlogp(x)/dx, multiply by 2.
        x_score = (x_score * 2.).cpu()

        # save each precomputed batch as a torch tensor file. these will be loaded and aggregated into a dataset later
        torch.save(x_score, os.path.join(SCORE_DIR, 'batch_{}.pt'.format(i)))
    

    print("Done computing scores...")

    ### IMPL GOAL 1

    # instantiate the CIFAR10 training dataset (wrapped in unshuffled dataloader of same batch size)
    # do not use augmentations other than ToTensor

    # together with the CIFAR10 training dataset, save the original image (plus score) in a channel-stacked format
    #       for example, for a cifar image x (C, H, W) = (3, 32, 32), save the cifar image with the score concatenated such that the result is (6, 32, 32)
    #       then, we can get x_image, x_score = x.chunk(2) once loaded

    #       we could try to save it in the same format as the original CIFAR10 dataset, but this *might* cause problems if we need a high-precision format for the scores
    #       it may be easiest to just save the dataset+scores as one big tensor. CIFAR10 isn't too big, so this may be ok to have in CPU memory
    
    # in another file (e.g., core/data/cifar10score), extend torchvision.datasets.CIFAR10 to load data from the above format.
    # add in logic (from train-wa.py args to gowal21uncovering/watrain.py training loop) to run an experiment with score projection training
    

    ### IMPL GOAL 2

    # with a corresponding command line arg, write a baseline experiment where our "score projection vector" is random.
    #       we would expect the AdvTrain adversarial attack to have zero cosine similarity with a random vector, so projected adversarial
    #       training with a random vector should yield the exact same result as Madry adversarial training
    #       
    #       This confirms that it is projection specifically along the EDM score direction, not just any subspace projection, which prevents performance loss


    


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


