#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0 python bayesian/train.py --model bayesian --save-dir outputs/bayesian \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test

SHTA=/data/deeplearning/CC/ShanghaiTech_Crowd_Counting_Dataset/part_A_final
SHTB=/data/deeplearning/CC/ShanghaiTech_Crowd_Counting_Dataset/part_B_final
UCF=/data/deeplearning/crowdcounting/UCF-Train-Val-Test

CUDA_VISIBLE_DEVICES=3 python bayesian/train.py --model bayesian --save-dir outputs/bayesian_joint \
                         --batch-size 8 \
                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
                         --device 3 \
                         --use-joint-dataset True \
                         --crop-size 512 \
                         --joint-dir ${SHTA},${SHTB},${UCF}


