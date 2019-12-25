#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python bayesian/test.py --save-dir outputs/bayesian/1225-001646 \
                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test

