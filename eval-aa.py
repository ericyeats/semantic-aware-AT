"""
Evaluation with AutoAttack.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from autoattack import AutoAttack
    
from core.data import get_data_info
from core.data import load_data
from core.models import create_model
from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed

from ce_hybrid_model import load_edm_from_path, load_cls_from_path, CEClassifier

SAMPLING_KWARGS = ['num_steps', 'sigma_min', 'sigma_max', 'rho', 'ce_sigma']



# Setup

parse = parser_eval()
# attack stuff (to fill in for classifier-->ce_classifier compatibility)
parse.add_argument('--attack', type=str, default='linf-pgd', choices=['linf-pgd', 'fgsm', 'linf-df'])
parse.add_argument('--attack_eps', type=float, default=0.031)
parse.add_argument('--log_path', type=str, default='./logs/last-eval.log')
parse.add_argument('--batch_size', type=int, default=2048)
parse.add_argument('--batch_size_aa', type=int, default=500)

parse.add_argument('--ce_classifier', action='store_true', help='Use a CE classifier')
parse.add_argument('--edm_pkl', type=str, default='https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl')
parse.add_argument('--cls_pt', type=str, default='./selection_model/cifar10_pseudo.pt')
parse.add_argument('--model', '-m', default='wrn-28-10', type=str,
                    help='name of the model')
parse.add_argument('--data', type=str, default='cifar10')
parse.add_argument('--n_ces', type=int, default=1)
parse.add_argument('--class_num', type=int, default=10)
parse.add_argument('--num_steps', type=int, default=100)
parse.add_argument('--sigma_min', type=float, default=0.002)
parse.add_argument('--sigma_max', type=float, default=80.0)
parse.add_argument('--rho', type=int, default=7)
parse.add_argument('--ce_sigma', type=float, default=0.2)
args = parse.parse_args()

samp_kwargs = {}
for key in args.__dict__:
    if key in SAMPLING_KWARGS:
        samp_kwargs[key] = args.__dict__[key]

if not args.ce_classifier:
    LOG_DIR = args.log_dir + '/' + args.desc
    with open(LOG_DIR+'/args.txt', 'r') as f:
        old = json.load(f)
        args.__dict__ = dict(vars(args), **old)

if args.data in ['cifar10', 'cifar10s']:
    da = '/cifar10/'
elif args.data in ['cifar100', 'cifar100s']:
    da = '/cifar100/'
elif args.data in ['svhn', 'svhns']:
    da = '/svhn/'
elif args.data in ['tiny-imagenet', 'tiny-imagenets']:
    da = '/tiny-imagenet/'


DATA_DIR = args.data_dir + da

if not args.ce_classifier:
    WEIGHTS = LOG_DIR + '/weights-best.pt'

    log_path = LOG_DIR + '/log-aa.log'
else:
    log_path = args.log_path

logger = Logger(log_path)
info = get_data_info(DATA_DIR)

# BATCH_SIZE = 32
# BATCH_SIZE_VALIDATION = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.log('Using device: {}'.format(device))


# Load data

seed(args.seed)
_, _, train_dataloader, test_dataloader = load_data(DATA_DIR, args.batch_size, args.batch_size, use_augmentation=False, 
                                                    shuffle_train=False)

if args.train:
    logger.log('Evaluating on training set.')
    l = [x for (x, y) in train_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in train_dataloader]
    y_test = torch.cat(l, 0)
else:
    l = [x for (x, y) in test_dataloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_dataloader]
    y_test = torch.cat(l, 0)

if args.num_samples is not None:
    x_test = x_test[:args.num_samples]
    y_test = y_test[:args.num_samples]

model = None

if args.ce_classifier:
    classifier = load_cls_from_path(args.cls_pt, args.model, class_num=args.class_num)
    edm = load_edm_from_path(args.edm_pkl)
    model = CEClassifier(edm, classifier, args.n_ces, **samp_kwargs)
    model = nn.DataParallel(model).cuda()
else:
    # Model
    print(args.model)
    model = create_model(args.model, args.normalize, info, device)
    checkpoint = torch.load(WEIGHTS)
    if 'tau' in args and args.tau:
        print ('Using WA model.')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    del checkpoint


# AA Evaluation

seed(args.seed)
norm = 'Linf' if args.attack in ['fgsm', 'linf-pgd', 'linf-df'] else 'L2'
adversary = AutoAttack(model, norm=norm, eps=args.attack_eps, log_path=log_path, version=args.version, seed=args.seed)

if args.version == 'custom':
    adversary.attacks_to_run = ['apgd-ce', 'apgd-t']
    adversary.apgd.n_restarts = 1
    adversary.apgd_targeted.n_restarts = 1

with torch.no_grad():
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=args.batch_size_aa)

print ('Script Completed.')