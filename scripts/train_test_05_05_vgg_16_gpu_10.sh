#!/bin/bash
#

MODEL_NAME=vgg_16
WEIGHT_LOSS=10
BATCH_SIZE=4
LEARNING_RATE=0.001
DECAY_TYPE=exponential
DECAY_FACTOR=0.97
DATA_NAME=kitti_calib
LOG_NAME=${DATA_NAME}_05_05
LIST_PARAM=5,0.5
LIDAR_POOL=4,2
SUMMARY_SECS=300
OPTIMIZER=momentum

EVAL_STEP=20000
for idx in {1..5}
do
    MAX_STEP=$(($idx*$EVAL_STEP))
    python deep_calib_train.py \
        --save_summaries_secs=${SUMMARY_SECS} \
        --optimizer=${OPTIMIZER} \
        --dataset_dir=/data/tf/${DATA_NAME} \
        --train_dir=/data/tf/checkpoints/${LOG_NAME} \
        --max_number_of_steps=${MAX_STEP} \
        --batch_size=${BATCH_SIZE} \
        --list_param=${LIST_PARAM} \
        --weight_loss=${WEIGHT_LOSS} \
        --lidar_pool=${LIDAR_POOL} \
        --model_name=${MODEL_NAME} \
        --checkpoint_path=pretrained/${MODEL_NAME}.ckpt \
        --checkpoint_exclude_scopes=${MODEL_NAME}/lidar_feat,${MODEL_NAME}/match_feat,${MODEL_NAME}/regression \
        --learning_rate_decay_type=${DECAY_TYPE} \
        --learning_rate=${LEARNING_RATE} \
        --ignore_missing_vars=True
        # --trainable_scopes=${MODEL_NAME}/lidar_feat,${MODEL_NAME}/match_feat,${MODEL_NAME}/regression \
        # --ignore_missing_vars=

    python deep_calib_test.py \
        --dataset_dir=/data/tf/${DATA_NAME} \
        --checkpoint_path=/data/tf/checkpoints/${LOG_NAME}/${MODEL_NAME}/weight_${WEIGHT_LOSS} \
        --list_param=${LIST_PARAM} \
        --lidar_pool=${LIDAR_POOL} \
        --model_name=${MODEL_NAME} \
        --weight_loss=${WEIGHT_LOSS}
done