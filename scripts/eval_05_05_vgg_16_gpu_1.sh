#!/bin/bash
#

MODEL_NAME=vgg_16
WEIGHT_LOSS=1
LOG_NAME=kitti_calib_05_05
LIST_PARAM=5,0.5
LIDAR_POOL=5,2

python deep_calib_kitti.py \
    --checkpoint_path=/data/tf/checkpoints/${LOG_NAME}/${MODEL_NAME}/weight_${WEIGHT_LOSS} \
    --list_param=${LIST_PARAM} \
    --lidar_pool=${LIDAR_POOL} \
    --model_name=${MODEL_NAME} \
    --weight_loss=${WEIGHT_LOSS} \
    --dir_image=data_ex/kitti/object/data_object_image_2/testing/image_2 \
    --dir_lidar=data_ex/kitti/object/data_object_velodyne/testing/velodyne \
    --dir_calib=data_ex/kitti/object/data_object_calib/testing/calib \
    --dir_out=data_ex/results/weight_${WEIGHT_LOSS}