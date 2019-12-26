#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=1 python cannet/train.py --model cannet --save-dir outputs/cannet \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 1

#CUDA_VISIBLE_DEVICES=0 python cannet/train.py --model oricannet --save-dir outputs/oricannet \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 0


#                         --val-epoch 1 \
#                         --val-start 0

#CUDA_VISIBLE_DEVICES=2 python asdnet/train.py --model asd --save-dir outputs/asd \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2 \
#                         --downsample-ratio 16

#CUDA_VISIBLE_DEVICES=2 python asdnet/train.py --model asd --save-dir outputs/asd \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2 \
#                         --downsample-ratio 16

#CUDA_VISIBLE_DEVICES=0 python asdnet/train.py --model scar --save-dir outputs/scar \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 0 \
#                         --downsample-ratio 8

CUDA_VISIBLE_DEVICES=1 python asdnet/train.py --model sfcn --save-dir outputs/sfcn \
                         --batch-size 8 \
                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
                         --device 1 \
                         --downsample-ratio 8


