# deep_calib
Calibration of sensors using deep learning

Structure of the code is motivated from [TensorFlow-Slim image classification library](https://github.com/tensorflow/models/tree/master/slim) and [Caffe](http://caffe.berkeleyvision.org/)

Some library functions use [TensorFlow-Slim image classification library](https://github.com/tensorflow/models/tree/master/slim) with and without modification.

## Generate KITTI Dataset for calibration in TfRecord Format
```
python tf_kitti_to_calib.py -gid {gpu IDs} -dir_in {Path to KITTI DATASET} -dir_out {Path to save tfrecord files}
```
ex)
```
python tf_kitti_to_calib.py -gid 0 -dir_in data/kitti -dir_out /data/tf/kitti_calib
```
#### Path to KITTI DATASET must have Object Detection data in following format (calib, image_2, velodyne)
```
{KITTI_PATH}/object/data_object_calib
{KITTI_PATH}/object/data_object_image_2
{KITTI_PATH}/object/data_object_velodyne
```
## Train using KITTI dataset.
First, you need to locate pretrained checkpoint file for each model.
We assumed that the files are located at the folder pretrained.
Pretrained models can be downloaded from the TensorFlow-Slim image classification library's [Pretrained models](https://github.com/tensorflow/models/tree/master/slim#Pretrained)

ex) VGG_16
Files located at:
```
{CURRENT_PATH}/pretrained/vgg_16.ckpt
```
Then you can run
```
CUDA_VISIBLE_DEVICES=1 ./scripts/train_vgg_16_gpu.sh
```
