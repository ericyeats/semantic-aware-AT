#!/bin/bash

python generate_ces.py --outdir=ce_data/out --base_indices=0-63 --batch=64 \
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
        --network_uncond=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
        --ce_sigma 0.2 --guidance 5 --perc 1.0 \
        --base_dataset_root ~/data --base_dataset cifar10 --n_ces 1 --steps 50


# torchrun --standalone --nproc_per_node 4 generate_ces.py \
#         --outdir out_cifar10_0_9999_ce1 --base_indices=0-9999 --batch 512 --class_num 10 \
#         --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
#         --network_uncond=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
#         --ce_sigma 0.05 --guidance 5. --perc 1.0 \
#         --base_dataset_root ~/data --base_dataset cifar10 --n_ces 1 --steps 50

        