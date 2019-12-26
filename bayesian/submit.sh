#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2 python bayesian/submit.py --model bayesian --save-dir outputs/bayesian/1225-001646 \
                         --data-dir /data/deeplearning/CC/test/A \
                         --sub-name bayesian_vgg19.csv

