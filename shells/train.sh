#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0 python bayesian/train.py --model bayesian --save-dir outputs/bayesian \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test

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

#CUDA_VISIBLE_DEVICES=1 python asdnet/train.py --model sfcn --save-dir outputs/sfcn \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 1 \
#                         --downsample-ratio 8

#CUDA_VISIBLE_DEVICES=1 python asdnet/train.py --model sfcn --save-dir outputs/sfcn \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 1 \
#                         --downsample-ratio 8

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

#CUDA_VISIBLE_DEVICES=0 python cannet/train.py --model oricannet --save-dir outputs/oricannet_warmup20_bs24 \
#                         --batch-size 24 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 0 \
#                         --lr 3e-4 \
#                         --warmup-epoch 20 \
#                         --steps 300,600 \
#                         --val-epoch 1 \
#                         --val-start 0

#CUDA_VISIBLE_DEVICES=3 python cannet/train.py --model sfcn --save-dir outputs/sfcn_warmup20_bs16 \
#                         --batch-size 12 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 3 \
#                         --lr 3e-4 \
#                         --warmup-epoch 20 \
#                         --steps 300,600 \
#                         --val-epoch 1 \
#                         --val-start 0

#CUDA_VISIBLE_DEVICES=3 python bayesian/train.py --model bayesian --save-dir outputs/bayesian_joint \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 3 \
#                         --use-joint-dataset True \
#                         --crop-size 512 \
#                         --joint-dir ${SHTA},${SHTB},${UCF}

#CUDA_VISIBLE_DEVICES=2 python bayesian/train.py --model sfanet --save-dir outputs/sfanet \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2

#CUDA_VISIBLE_DEVICES=3 python bayesian/train.py --model bayesian --save-dir outputs/bayesian_joint_scale \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 3 \
#                         --use-joint-dataset True \
#                         --crop-size 512 \
#                         --joint-dir ${SHTA},${SHTB},${UCF}

#CUDA_VISIBLE_DEVICES=2 python bayesian/train.py --model bayesian --save-dir outputs/bayesian_warmup20_bs24 \
#                         --batch-size 24 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2 \
#                         --lr 3e-4 \
#                         --warmup-epoch 20 \
#                         --steps 300,600 \
#                         --val-epoch 1 \
#                         --val-start 0 \

#CUDA_VISIBLE_DEVICES=0 python shells/train.py --model oricannet --save-dir outputs/oricannet_bs16 \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 0 \
#                         --val-epoch 1 \
#                         --val-start 0

#CUDA_VISIBLE_DEVICES=0 python shells/train.py --model oricannet --save-dir outputs/oricannet_bs16 \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 0 \
#                         --val-epoch 1 \
#                         --val-start 0

#CUDA_VISIBLE_DEVICES=3 python shells/train.py --model oricannet --save-dir outputs/oricannet_bs16_joint \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 3 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --use-joint-dataset True \
#                         --joint-dir ${SHTA},${SHTB},${UCF}

#CUDA_VISIBLE_DEVICES=2,3 python shells/train.py --model c3f_sanet --save-dir outputs/c3f_sanet_bs16 \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2,3 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --downsample-ratio 1

#CUDA_VISIBLE_DEVICES=1 python shells/train.py --model c3f_csrnet --save-dir outputs/c3f_csrnet_bs16 \
#                         --batch-size 4 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 1 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --downsample-ratio 1

#CUDA_VISIBLE_DEVICES=1 python -u shells/train.py --model c3f_res101_sfcn --save-dir outputs/c3f_res101_sfcn_bs16 \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 1 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --downsample-ratio 8

#CUDA_VISIBLE_DEVICES=3 python -u shells/train.py --model res50_fpn --save-dir outputs/res50_fpn_bs16 \
#                         --batch-size 4 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 3 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --downsample-ratio 1

#CUDA_VISIBLE_DEVICES=3 python -u shells/train.py --model scar --save-dir outputs/scar_bs16 \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 3 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --downsample-ratio 8

#-----------------------------------------------Mutil GPU-----------------------------------------------#

#CUDA_VISIBLE_DEVICES=0,1 python shells/train.py --model oricannet --save-dir outputs/oricannet_mutilgpu \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 0,1 \
#                         --val-epoch 1 \
#                         --val-start 0

#CUDA_VISIBLE_DEVICES=2,3 python shells/train.py --model bayesian --save-dir outputs/bayesian_mutilgpu \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2,3 \
#                         --val-epoch 1 \
#                         --val-start 0

#-----------------------------------------------Single GPU-----------------------------------------------#

#CUDA_VISIBLE_DEVICES=2 python shells/train.py --model bayesian --save-dir outputs/bayesian \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2 \
#                         --val-epoch 1 \
#                         --val-start 0

#CUDA_VISIBLE_DEVICES=3 python shells/train.py --model oricannet --save-dir outputs/oricannet \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 3 \
#                         --val-epoch 1 \
#                         --val-start 0

#CUDA_VISIBLE_DEVICES=1 python shells/train.py --model c3f_csrnet --save-dir outputs/c3f_csrnet \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 1 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --downsample-ratio 8

#CUDA_VISIBLE_DEVICES=0 python shells/train.py --model sfcn --save-dir outputs/sfcn \
#                         --batch-size 12 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 0 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --downsample-ratio 8

CUDA_VISIBLE_DEVICES=2 python shells/train.py --model oricannet --save-dir outputs/oricannet_steplr \
                         --batch-size 16 \
                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
                         --device 2 \
                         --val-epoch 1 \
                         --val-start 0 \
                         --steps 300,600
