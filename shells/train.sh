#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0 python bayesian/train.py --model bayesian --save-dir outputs/bayesian \
#                         --batch-size 8 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test

SHTA=/data/deeplearning/crowdcounting/SHTA-Train-Val-Test
SHTB=/data/deeplearning/crowdcounting/SHTB-Train-Val-Test
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

#CUDA_VISIBLE_DEVICES=2 python shells/train.py --model oricannet --save-dir outputs/oricannet_steplr \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --steps 300,600

#-----------------------------------------------Resume-----------------------------------------------#

#CUDA_VISIBLE_DEVICES=1 python shells/train.py --model oricannet --save-dir outputs/oricannet_steplr_jointdataset \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 1 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --steps 300,600 \
#                         --use-joint-dataset True \
#                         --joint-dir ${SHTA},${SHTB},${UCF} \
#                         --resume outputs/oricannet_steplr_jointdataset/1229-141742/6_ckpt.tar

#CUDA_VISIBLE_DEVICES=2 python shells/train.py --model oricannet --save-dir outputs/oricannet_steplr \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --steps 300,600 \
#                         --resume outputs/oricannet_steplr/1229-121332/77_ckpt.tar

#CUDA_VISIBLE_DEVICES=3 python shells/train.py --model oricannet --save-dir outputs/oricannet \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 3 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --resume outputs/oricannet/1228-122712/898_ckpt.tar \
#                         --steps 300,600

#CUDA_VISIBLE_DEVICES=0 python shells/train.py --model sfcn --save-dir outputs/sfcn \
#                         --batch-size 12 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 0 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --downsample-ratio 8 \
#                         --resume outputs/sfcn/1228-131623/501_ckpt.tar \
#                         --steps 300,600

#-----------------------------------------------Mutil Steps-----------------------------------------------#

#CUDA_VISIBLE_DEVICES=2 python shells/train.py --model oricannet --save-dir outputs/oricannet_steplr \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --steps 300,600

#CUDA_VISIBLE_DEVICES=1 python shells/train.py --model oricannet --save-dir outputs/oricannet_steplr_jointdataset \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 1 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --steps 300,600 \
#                         --use-joint-dataset True \
#                         --joint-dir ${SHTA},${SHTB},${UCF}

#CUDA_VISIBLE_DEVICES=0 python shells/train.py --model sfcn --save-dir outputs/sfcn \
#                         --batch-size 12 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 0 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --downsample-ratio 8 \
#                         --steps 300,600

#CUDA_VISIBLE_DEVICES=3 python shells/train.py --model oricannetvgg19 --save-dir outputs/oricannetvgg19 \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 3 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --downsample-ratio 8 \
#                         --steps 300,600

#CUDA_VISIBLE_DEVICES=2 python shells/train.py --model oricannet --save-dir outputs/oricannet_steplr \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 2 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --steps 300,600 \
#                         --resume outputs/oricannet_steplr/1229-151811/654_ckpt.tar

#CUDA_VISIBLE_DEVICES=1 python shells/train.py --model oricannet --save-dir outputs/oricannet_pretrain_ucf_train_shanghaitechA \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 1 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --use-joint-dataset True \
#                         --joint-dir ${SHTA},${UCF} \
#                         --resume outputs/oricannet/1228-122712/best_model.pth

#CUDA_VISIBLE_DEVICES=3 python shells/train.py --model oricannet --save-dir outputs/oricannet_trainval \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 3 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --trainval True

#CUDA_VISIBLE_DEVICES=3 python shells/train.py --model oricannet --save-dir outputs/oricannet_lr1e-4 \
#                         --batch-size 16 \
#                         --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test \
#                         --device 3 \
#                         --val-epoch 1 \
#                         --steps 700 \
#                         --lr 1e-4 \
#                         --val-start 0

#CUDA_VISIBLE_DEVICES=2 python shells/train.py --model oricannet --save-dir outputs/oricannet_trainval \
#                         --batch-size 16 \
#                         --data-dir ${UCF} \
#                         --device 2 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --trainval True

#CUDA_VISIBLE_DEVICES=1 python shells/train.py --model oricannet --save-dir outputs/oricannet_joint \
#                         --batch-size 16 \
#                         --data-dir ${SHTA},${SHTB},${UCF} \
#                         --device 1 \
#                         --val-epoch 1 \
#                         --val-start 0 \
#                         --use-joint-dataset True

CUDA_VISIBLE_DEVICES=0 python shells/train.py --model sfcn --save-dir outputs/sfcn_trainval \
                         --batch-size 12 \
                         --data-dir ${UCF} \
                         --device 0 \
                         --val-epoch 1 \
                         --val-start 0 \
                         --trainval True