#!/usr/bin/env bash
python bayesian/preprocess_dataset.py --origin-dir /data/deeplearning/crowdcounting/UCF-QNRF_ECCV18 \
                                      --txt-dir crowd_counting_pytorch/datasets/ucf \
                                      --data-dir /data/deeplearning/crowdcounting/UCF-Train-Val-Test

