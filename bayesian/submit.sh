#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0,1 python bayesian/submit.py --save-dir outputs/bayesian/1225-001646 \
                         --data-dir /data/deeplearning/CC/test/A \
                         --sub-name bayesian_vgg19.csv

