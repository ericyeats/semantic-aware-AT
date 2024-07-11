#!/bin/bash

# python train-wa.py --data-dir ~/data \
#     --log-dir 'trained_models' \
#     --desc 'WRN28-10Swish_cifar10_lr0p2_MADRY_epoch100_bs512' \
#     --data cifar10 \
#     --batch-size 512 \
#     --batch-size-validation 512 \
#     --model wrn-28-10-swish \
#     --num-adv-epochs 100 \
#     --lr 0.2

# python train-wa.py --data-dir ~/data \
#     --log-dir 'trained_models' \
#     --desc 'WRN28-10Swish_cifar10_lr0p2_STANDARD_epoch100_bs512' \
#     --data cifar10 \
#     --batch-size 512 \
#     --batch-size-validation 512 \
#     --model wrn-28-10-swish \
#     --num-adv-epochs 100 \
#     --lr 0.2 \
#     --standard

python train-wa.py --data-dir ~/data \
    --log-dir 'trained_models' \
    --desc 'WRN28-10Swish_cifar10_lr0p2_TRADESLSE_epoch100_bs512' \
    --data cifar10 \
    --batch-size 512 \
    --batch-size-validation 512 \
    --model wrn-28-10-swish \
    --num-adv-epochs 100 \
    --lr 0.2 \
    --beta 5.0 \
    --LSE


# python train-wa.py --data-dir ./score_data \
#     --log-dir 'trained_models' \
#     --desc 'WRN28-10Swish_cifar10_lr0p2_SPTRADESLSE_g0p0_n40_t0p2_epoch100_bs512' \
#     --data cifar10score \
#     --beta 5.0 \
#     --LSE \
#     --batch-size 512 \
#     --batch-size-validation 512 \
#     --time 0.2 \
#     --n_mc_samples 40 \
#     --gamma 0.0 \
#     --model wrn-28-10-swish \
#     --num-adv-epochs 100 \
#     --lr 0.2



    # --score_network_pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl


