#!/bin/bash
#

MODEL_NAME=vgg_16
WEIGHT_LOSS=10

python deep_calib_train.py \
    --dataset_dir=data_ex/tf/kitti_calib \
    --train_dir=checkpoints/kitti_calib \
    --max_number_of_steps=500 \
    --list_param=20,1.5 \
    --weight_loss=${WEIGHT_LOSS} \
    --clone_on_cpu=True \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=pretrained/${MODEL_NAME}.ckpt \
    --checkpoint_exclude_scopes=${MODEL_NAME}/lidar_feat,${MODEL_NAME}/match_feat,${MODEL_NAME}/regression
    # --trainable_scopes=
    # --ignore_missing_vars=

python deep_calib_test.py \
    --dataset_dir=data_ex/tf/kitti_calib \
    --checkpoint_path=checkpoints/kitti_calib/${MODEL_NAME}/weight_${WEIGHT_LOSS} \
    --list_param=20,1.5 \
    --model_name=${MODEL_NAME} \
    --weight_loss=${WEIGHT_LOSS}