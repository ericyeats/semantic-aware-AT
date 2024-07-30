#!/bin/bash

python generate_ces_uncond.py --outdir=ce_data/out --base_indices=0-63 --batch=64 \
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
        --ce_sigma 0.2 \
        --base_dataset_root ~/data --base_dataset cifar10 --n_ces 1 --steps 100

python visualize_samples.py ./ce_data/out/0.npz

# torchrun --standalone --nproc_per_node 4 generate_ces_uncond.py \
#         --outdir ce_data/out_cifar10_0_49999_ce5 --base_indices=0-49999 --batch 512 \
#         --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
#         --ce_sigma 0.2 --n_ces 5 \
#         --base_dataset_root ~/data --base_dataset cifar10 --steps 50

# torchrun --standalone --nproc_per_node 4 generate_ces_uncond.py \
#         --outdir ce_data/out_cifar10_test_0_9999_ce5 --base_indices=0-9999 --batch 512 \
#         --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-uncond-vp.pkl \
#         --ce_sigma 0.2 --n_ces 5 --test \
#         --base_dataset_root ~/data --base_dataset cifar10 --steps 50