#!/bin/bash
#

MODEL_NAME=vgg_16
WEIGHT_LOSS=10
DATA_NAME=kitti_calib_10_10
LIST_PARAM=10,1.0
LIDAR_POOL=5,2

python deep_calib_kitti.py \
    --dataset_dir=/data/tf/${DATA_NAME} \
    --checkpoint_path=pretrained \
    --list_param=${LIST_PARAM} \
    --lidar_pool=${LIDAR_POOL} \
    --model_name=${MODEL_NAME} \
    --weight_loss=${WEIGHT_LOSS} \
    --dir_image=data_ex/kitti/object/data_object_image_2/testing/image_2 \
    --dir_lidar=data_ex/kitti/object/data_object_velodyne/testing/velodyne \
    --dir_calib=data_ex/kitti/object/data_object_calib/testing/calib \
    --dir_out=data_ex/results/weight_${WEIGHT_LOSS}