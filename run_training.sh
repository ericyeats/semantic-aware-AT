#!/bin/bash

python train-wa.py --data-dir ~/data \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10_lr0p2_MADRY_epoch100_bs128' \
    --data cifar10 \
    --batch-size 128 \
    --model wrn-28-10-swish \
    --num-adv-epochs 100 \
    --lr 0.2

# python train-wa.py --data-dir ~/data \
#     --log-dir 'trained_models' \
#     --desc 'WRN28-10Swish_cifar10_lr0p2_CARAT_g0p3_n10_t0p1_epoch100_bs128' \
#     --data cifar10 \
#     --batch-size 128 \
#     --model wrn-28-10-swish \
#     --num-adv-epochs 100 \
#     --lr 0.2 \
#     --score \
#     --score_network_pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl


