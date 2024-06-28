#!/bin/bash

python train-wa.py --data-dir ~/data \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10_lr0p2_MADRY_epoch100_bs512' \
    --data cifar10 \
    --batch-size 512 \
    --batch-size-validation 512 \
    --model wrn-28-10-swish \
    --num-adv-epochs 100 \
    --lr 0.2

# python train-wa.py --data-dir ~/data \
#     --log-dir 'trained_models' \
#     --desc 'WRN28-10Swish_cifar10_lr0p2_CARAT_g0p5_n40_t0p1_epoch100_bs512' \
#     --data cifar10 \
#     --batch-size 512 \
#     --batch-size-validation 512 \
#     --n_mc_samples 40 \
#     --n_chunks 10 \
#     --model wrn-28-10-swish \
#     --num-adv-epochs 100 \
#     --lr 0.2 \
#     --score \
#     --score_network_pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl


