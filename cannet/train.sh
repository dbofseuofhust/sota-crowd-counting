#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python cannet/train.py --model cannet --save-dir outputs/cannet \
                         --batch-size 8 \
                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
                         --device 1

