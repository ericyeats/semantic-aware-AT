#!/bin/bash

python -u record_scores.py --data-dir '~/data' \
    --data 'cifar10' \
    --batch-size 512 \
    --score_network_pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --time 0.01 \
    --n_mc_samples 40 \
    --n_chunks 10

python -u record_scores.py --data-dir '~/data' \
    --data 'cifar10' \
    --batch-size 512 \
    --score_network_pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --time 0.05 \
    --n_mc_samples 40 \
    --n_chunks 10

python -u record_scores.py --data-dir '~/data' \
    --data 'cifar10' \
    --batch-size 512 \
    --score_network_pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --time 0.1 \
    --n_mc_samples 40 \
    --n_chunks 10

python -u record_scores.py --data-dir '~/data' \
    --data 'cifar10' \
    --batch-size 512 \
    --score_network_pkl https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl \
    --time 0.2 \
    --n_mc_samples 40 \
    --n_chunks 10