#!/bin/bash
#

MODEL_NAME=vgg_16
WEIGHT_LOSS=10
BATCH_SIZE=32
LEARNING_RATE=0.00000001
END_LEARNING_RATE=0.0000000001
DATA_NAME=kitti_calib_small
LIST_PARAM=5,0.5
LIDAR_POOL=5,2
SUMMARY_SECS=180

python deep_calib_train.py \
    --save_summaries_secs=${SUMMARY_SECS} \
    --dataset_dir=/data/tf/${DATA_NAME} \
    --train_dir=/data/tf/checkpoints/${DATA_NAME} \
    --max_number_of_steps=80000 \
    --batch_size=${BATCH_SIZE} \
    --list_param=${LIST_PARAM} \
    --weight_loss=${WEIGHT_LOSS} \
    --lidar_pool=${LIDAR_POOL} \
    --clone_on_cpu=False \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=pretrained/${MODEL_NAME}.ckpt \
    --checkpoint_exclude_scopes=${MODEL_NAME}/lidar_feat,${MODEL_NAME}/match_feat,${MODEL_NAME}/regression \
    # --trainable_scopes=${MODEL_NAME}/lidar_feat,${MODEL_NAME}/match_feat,${MODEL_NAME}/regression \
    --learning_rate=${LEARNING_RATE} \
    --end_learning_rate=${END_LEARNING_RATE}
    # --ignore_missing_vars=

python deep_calib_test.py \
    --dataset_dir=/data/tf/${DATA_NAME} \
    --checkpoint_path=/data/tf/checkpoints/${DATA_NAME}/${MODEL_NAME}/weight_${WEIGHT_LOSS} \
    --list_param=${LIST_PARAM} \
    --lidar_pool=${LIDAR_POOL} \
    --model_name=${MODEL_NAME} \
    --weight_loss=${WEIGHT_LOSS}