#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0 python bayesian/submit.py --model bayesian --save-dir outputs/bayesian/1225-001646 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name bayesian_640_vgg19.csv

#CUDA_VISIBLE_DEVICES=3 python bayesian/submit.py --model bayesian --save-dir outputs/bayesian_joint/1225-234557 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name bayesian_joint_noscale_vgg19.csv

CUDA_VISIBLE_DEVICES=1 python bayesian/submit.py --save-dir /home/dongbin/DeepBlueAI/sota-crowd-counting/outputs/bayesian \
                         --data-dir /data/deeplearning/CC/test/A \
                         --sub-name ori_bayesian_vgg19.csv
