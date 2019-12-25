#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python cannet/train.py --save-dir outputs/cannet \
                         --batch-size 8 \
                         --data-dir /data/deeplearning/CC/ShanghaiTech_Crowd_Counting_Dataset \
                         --lr 1e-7
                         

