#!/usr/bin/env bash

# test:
# 512: Final Test: mae 90.73611527859808, mse 161.14400647847842
# 544: Final Test: mae 90.73611527859808, mse 161.14400647847842
CUDA_VISIBLE_DEVICES=0 python bayesian/test.py --model bayesian --save-dir outputs/bayesian/1225-001646 \
                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test

