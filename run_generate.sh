#!/bin/bash

# Generate 10K images for each class using 4 V100 GPUs, using deterministic sampling with 20 steps
torchrun --standalone --nproc_per_node=4 generate.py --outdir=out_cifar10_0_9999 --seeds=0-9999 --batch=512  --steps=50 --class_num=10 \
    --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl