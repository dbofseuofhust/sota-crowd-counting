#!/usr/bin/env bash

SHTA=/data/deeplearning/CC/ShanghaiTech_Crowd_Counting_Dataset/part_A_final
SHTB=/data/deeplearning/CC/ShanghaiTech_Crowd_Counting_Dataset/part_B_final
UCF=/data/deeplearning/crowdcounting/UCF-Train-Val-Test

#CUDA_VISIBLE_DEVICES=1 python cannet/train.py --model cannet --save-dir outputs/cannet \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 1

#CUDA_VISIBLE_DEVICES=0 python cannet/train.py --model oricannet --save-dir outputs/oricannet \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 0 \

#CUDA_VISIBLE_DEVICES=2 python cannet/train.py --model cannet --save-dir outputs/cannet_joint \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2 \
#                         --use-joint-dataset True \
#                         --crop-size 512 \
#                         --joint-dir ${SHTA},${SHTB},${UCF}

#CUDA_VISIBLE_DEVICES=2 python cannet/train.py --model asd --save-dir outputs/asd \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2 \
