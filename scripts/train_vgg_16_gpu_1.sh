#!/bin/bash
#

MODEL_NAME=vgg_16

python deep_calib_train.py \
    --dataset_dir=/data/tf/kitti_calib \
    --train_dir=checkpoints/kitti_calib/${MODEL_NAME}/weight1 \
    --max_number_of_steps=5000 \
    --list_param=20,1.5 \
    --clone_on_cpu=False \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=pretrained/${MODEL_NAME}.ckpt \
    --checkpoint_exclude_scopes=${MODEL_NAME}/lidar_feat,${MODEL_NAME}/match_feat,${MODEL_NAME}/regression
    # --trainable_scopes=
    # --ignore_missing_vars=