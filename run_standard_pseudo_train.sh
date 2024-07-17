#!/bin/bash

python train-wa.py --data-dir ~/data \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10s_Syn30k_lr0p2_MADRY_PSEUDO_epoch100_bs512_fraction0p2_ls0p1' \
    --data cifar10s \
    --batch-size 512 \
    --batch-size-validation 512 \
    --model wrn-28-10-swish \
    --num-adv-epochs 100 \
    --lr 0.2 \
    --ls 0.1 \
    --aux-data-filename './synthetic_data/cifar10_30k/dataset.npz' \
    --unsup-fraction 0.2 \
    --standard_pseudo