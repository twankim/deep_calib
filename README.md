# deep_calib
Calibration of sensors using deep learning

## Generate KITTI Dataset for calibration in TfRecord Format
```
python tf_kitti_to_calib.py -gid {gpu IDs} -dir_in {Path to KITTI DATASET} -dir_out {Path to save tfrecord files}
```

#### Path to KITTI DATASET must have Object Detection data in following format (calib, image_2, velodyne)
```
   {KITTI_PATH}/object/data_object_calib
   {KITTI_PATH}/object/data_object_image_2
   {KITTI_PATH}/object/data_object_velodyne
```
