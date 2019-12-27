#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0 python bayesian/train.py --model bayesian --save-dir outputs/bayesian \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test

SHTA=/data/deeplearning/CC/ShanghaiTech_Crowd_Counting_Dataset/part_A_final
SHTB=/data/deeplearning/CC/ShanghaiTech_Crowd_Counting_Dataset/part_B_final
UCF=/data/deeplearning/crowdcounting/UCF-Train-Val-Test

CUDA_VISIBLE_DEVICES=2 python bayesian/train.py --save-dir outputs/oribayesian \
                         --batch-size 8 \
                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
                         --device 2 \
