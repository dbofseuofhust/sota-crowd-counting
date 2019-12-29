#!/usr/bin/env bash
#CUDA_VISIBLE_DEVICES=0 python bayesian/submit.py --model bayesian --save-dir outputs/bayesian/1225-001646 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name bayesian_640_vgg19.csv

#CUDA_VISIBLE_DEVICES=3 python bayesian/submit.py --model bayesian --save-dir outputs/bayesian_joint/1225-234557 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name bayesian_joint_noscale_vgg19.csv

#CUDA_VISIBLE_DEVICES=0 python bayesian/submit.py --model bayesian --save-dir outputs/bayesian_joint_scale/1227-001358 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name bayesian_joint_scale_vgg19.csv

#CUDA_VISIBLE_DEVICES=0 python bayesian/submit.py --model bayesian --save-dir outputs/bayesian_joint_scale/1227-001358 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name bayesian_joint_scale_vgg19.csv

#CUDA_VISIBLE_DEVICES=0 python shells/submit.py --model bayesian --save-dir outputs/before/bayesian/1225-001646 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name bayesian_full_scale_vgg19.csv

#CUDA_VISIBLE_DEVICES=0 python shells/submit.py --model bayesian --save-dir outputs/before/bayesian/1225-001646 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name bayesian_full_scale_vgg19.csv

#CUDA_VISIBLE_DEVICES=0 python shells/submit.py --model oricannet --save-dir outputs/before/oricannet/1225-220629 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name oricannet_full_scale_vgg19.csv

#CUDA_VISIBLE_DEVICES=0 python shells/submit.py --model sfcn --save-dir outputs/before/sfcn/1226-211706 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name sfcn_full_scale_vgg19.csv

#CUDA_VISIBLE_DEVICES=0 python shells/submit.py --model bayesian --save-dir outputs/before/bayesian_joint_scale/1227-001358 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name bayesian_joint_scale_full_scale_vgg19.csv

#CUDA_VISIBLE_DEVICES=1 python shells/submit.py --model oricannet --save-dir outputs/oricannet/1228-122712 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name fine_oricannet_scale_full_scale_vgg19.csv

#CUDA_VISIBLE_DEVICES=1 python shells/submit.py --model sfcn --save-dir outputs/sfcn/1228-131623 \
#                         --data-dir /data/deeplearning/CC/test/A \
#                         --sub-name fine_sfcn_scale_full_scale_vgg19.csv

CUDA_VISIBLE_DEVICES=1 python shells/submit.py --model c3f_csrnet --save-dir outputs/c3f_csrnet/1228-123254 \
                         --data-dir /data/deeplearning/CC/test/A \
                         --sub-name fine_c3f_csrnet_scale_full_scale_vgg19.csv