# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-05 13:32:38
# @Last Modified by:   twankim
# @Last Modified time: 2017-09-27 15:40:12

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets.config import cfg

_NAME_DATA = 'kitti'

_NUM_PREDS = {'num_preds':[4,3],
              'is_normalize':[True,False]
              }

_NUM_TRAIN = 7481
_NUM_TEST = 7518

_NUM_SAMPLES = {'train': _NUM_TRAIN,
                'test': _NUM_TEST*cfg._NUM_GEN}

def convert_calib_mat(vid,cal_val):
    assert vid in cfg._SET_CALIB, '!! Wrong parsing occurred: {}'.format(vid)
    if vid == cfg._SET_CALIB[0]: # P2
        return cal_val.reshape((3,4))
    elif vid == cfg._SET_CALIB[1]: # R0_rect
        cal_mat = np.zeros((4,4))
        cal_mat[:3,:3] = cal_val.reshape((3,3))
        cal_mat[3,3] = 1
        return cal_mat
    else: # Tr_velo_to_cam
        cal_mat = np.zeros((4,4))
        cal_mat[:3,:4] = cal_val.reshape((3,4))
        cal_mat[3,3] = 1
        return cal_mat

# Read & Save calibration matrix
def get_calib_mat(f_calib):
    dict_calib = {}
    with open(f_calib,'r') as input_file:
        for line in input_file:
            if len(line) > 1:
                vid,vals = line.split(':',1)
                val_cal = [float(v) for v in vals.split()]
                if vid in cfg._SET_CALIB:
                    dict_calib[vid] = convert_calib_mat(vid,np.array(val_cal))
    return dict_calib

def get_data(path_data,image_set,list_param=[],reader=None):
    """ Returns a dataset

    Args:
        path_data: path to dataset (including dir)
        image_set: String, split name (train/test)
        list_param: list of parameters in the TFRecord file's name
        reader: The subclass of tf.ReaderBase. If left as `None`, 
                then the default reader defined by each dataset is used.

    Returns:
        A 'Dataset' class.
    """
    assert image_set in _NUM_SAMPLES,\
            "!! {} data is not supported".format(image_set)
    if reader is None:
        reader = tf.TFRecordReader

    if image_set == 'train':
        path_file = os.path.join(path_data,
                                 cfg._TF_FORMAT_TRAIN.format(_NAME_DATA,image_set))

        keys_to_features = {
            'image/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature(
                (), tf.string, default_value='png'),
            'lidar/points': tf.VarLenFeature(
                tf.float32),
            'calib/mat_intrinsic': tf.FixedLenFeature(
                [12], tf.float32, default_value=tf.zeros([12], dtype=tf.float32)),
            'calib/mat_rect': tf.FixedLenFeature(
                [16], tf.float32, default_value=tf.zeros([16], dtype=tf.float32)),
            'calib/mat_extrinsic': tf.FixedLenFeature(
                [16], tf.float32, default_value=tf.zeros([16], dtype=tf.float32)),
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(image_key='image/encoded',
                                                  format_key='image/format',
                                                  channels=3),
            'points': slim.tfexample_decoder.Tensor('lidar/points'),
            'mat_intrinsic': slim.tfexample_decoder.Tensor('calib/mat_intrinsic'),
            'mat_rect': slim.tfexample_decoder.Tensor('calib/mat_rect'),
            'mat_extrinsic': slim.tfexample_decoder.Tensor('calib/mat_extrinsic')
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(
                                keys_to_features,items_to_handlers)

        return slim.dataset.Dataset(
                data_sources=path_file,
                reader=reader,
                decoder=decoder,
                num_samples=_NUM_SAMPLES[image_set],
                num_preds=_NUM_PREDS,
                items_to_descriptions=None)

    elif image_set == 'test':
        list_param[0] = int(list_param[0])
        list_param[1] = float(list_param[1])
        list_param.insert(0,_NAME_DATA)
        list_param.insert(1,image_set)
        path_file = os.path.join(path_data,
                                 cfg._TF_FORMAT_TEST.format(*list_param))

        keys_to_features = {
            'image/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'image/format': tf.FixedLenFeature(
                (), tf.string, default_value='png'),
            'lidar/encoded': tf.FixedLenFeature(
                (), tf.string, default_value=''),
            'param/y_calib': tf.FixedLenFeature(
                [7], tf.float32, default_value=tf.zeros([7], dtype=tf.float32)),
            'param/rot_angle': tf.FixedLenFeature(
                [1], tf.float32, default_value=tf.zeros([1], dtype=tf.float32)),
            'param/a_vec': tf.FixedLenFeature(
                [3], tf.float32, default_value=tf.zeros([3], dtype=tf.float32)),
            'param/params_crop': tf.FixedLenFeature(
                [4], tf.int64, default_value=tf.zeros([4], dtype=tf.int64))
        }

        items_to_handlers = {
            'image': slim.tfexample_decoder.Image(image_key='image/encoded',
                                                  format_key='image/format',
                                                  channels=3),
            'lidar': slim.tfexample_decoder.Image(image_key='lidar/encoded',
                                                  format_key='image/format',
                                                  channels=1),
            'y': slim.tfexample_decoder.Tensor('param/y_calib'),
            'theta': slim.tfexample_decoder.Tensor('param/rot_angle'),
            'a_vec': slim.tfexample_decoder.Tensor('param/a_vec'),
            'params_crop': slim.tfexample_decoder.Tensor('param/params_crop')
        }

        decoder = slim.tfexample_decoder.TFExampleDecoder(
                                keys_to_features,items_to_handlers)

        return slim.dataset.Dataset(
                data_sources=path_file,
                reader=reader,
                decoder=decoder,
                num_samples=_NUM_SAMPLES[image_set],
                num_preds=_NUM_PREDS,
                items_to_descriptions=None)
