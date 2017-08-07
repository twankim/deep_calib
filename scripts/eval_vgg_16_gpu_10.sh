#!/bin/bash
#

MODEL_NAME=vgg_16
WEIGHT_LOSS=10

python deep_calib_kitti.py \
    --dataset_dir=/data/tf/kitti_calib \
    --checkpoint_path=/data/tf/checkpoints/kitti_calib/${MODEL_NAME}/weight_${WEIGHT_LOSS} \
    --list_param=20,1.5 \
    --model_name=${MODEL_NAME} \
    --weight_loss=${WEIGHT_LOSS} \
    --dir_image=data_ex/kitti/object/data_object_image_2/testing/image_2 \
    --dir_lidar=data_ex/kitti/object/data_object_velodyne/testing/velodyne \
    --dir_calib=data_ex/kitti/object/data_object_calib/testing/calib \
    --dir_out=data_ex/results/weight_${WEIGHT_LOSS}