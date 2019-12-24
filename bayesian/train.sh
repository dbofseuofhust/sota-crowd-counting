#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python bayesian/train.py --save-dir outputs/bayesian \
                         --batch-size 8 \
                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test

