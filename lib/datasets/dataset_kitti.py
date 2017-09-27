# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-05 13:32:38
# @Last Modified by:   twankim
# @Last Modified time: 2017-09-27 09:02:09

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

_D_MAX = 50.0
_D_MIN = 0.5

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

def coord_transform(points, t_mat):
    # Change to homogeneous form
    points = np.hstack([points,np.ones((np.shape(points)[0],1))])
    t_points = np.dot(points,t_mat.T)
    # Normalize
    t_points = t_points[:,:-1]/t_points[:,[-1]]
    return t_points

def project_lidar_to_img(dict_calib,points,im_height,im_width):
    # Extract depth data first before projection to 2d image space
    trans_mat = np.dot(dict_calib[cfg._SET_CALIB[1]],dict_calib[cfg._SET_CALIB[2]])
    points3D = coord_transform(points,trans_mat)
    pointsDist = points3D[:,2]

    # Project to image space
    trans_mat = np.dot(dict_calib[cfg._SET_CALIB[0]],trans_mat)
    points2D = coord_transform(points,trans_mat)

    # Find only feasible points
    idx1 = (points2D[:,0]>=0) & (points2D[:,0] <=im_width-1)
    idx2 = (points2D[:,1]>=0) & (points2D[:,1] <=im_height-1)
    idx3 = (pointsDist>=0)
    idx_in = idx1 & idx2 & idx3
    points2D_fin = points2D[idx_in,:]
    pointsDist_fin = pointsDist[idx_in]

    return points2D_fin, pointsDist_fin

def dist_to_pixel(val_dist, mode='inverse', d_max=_D_MAX, d_min=_D_MIN):
    """ Returns pixel value from distance measurment
    Args:
        val_dist: distance value (m)
        mode: 'inverse' vs 'standard'
        d_max: maximum distance to consider
        d_min: minimum distance to consider
    Returns:
        pixel value in 'uint8' format
    """
    val_dist = d_max if val_dist>d_max else val_dist if val_dist>d_min else d_min
    if mode == 'standard':
        return np.round(val_dist*255.0/d_max).astype('uint8')
    elif mode == 'inverse':
        return np.round(d_min*255.0/val_dist).astype('uint8')
    else:
        # Default is inverse
        return np.round(d_min*255.0/val_dist).astype('uint8')

def points_to_img(points2D,pointsDist,im_height,im_width):
    im_depth = np.zeros((im_height,im_width),dtype=np.uint8)
    for i in xrange(np.shape(points2D)[0]):
        x,y = np.round(points2D[i,:]).astype('int')
        im_depth[y,x] = dist_to_pixel(pointsDist[i])
        # im_depth[y,x] = dist_to_pixel(pointsDist[i],mode='standard')
    return im_depth

def tf_coord_transform(points, t_mat):
    # Change to homogeneous form
    points = tf.concat([points,tf.ones([tf.shape(points)[0],1],tf.float32)], 1)
    t_points = tf.matmul(points,tf.transpose(t_mat))
    # Normalize
    t_points = tf.div(t_points[:,:-1],tf.expand_dims(t_points[:,-1],1))
    return t_points

def tf_project_lidar_to_img(dict_calib,points,im_height,im_width):
    # Extract depth data first before projection to 2d image space
    trans_mat = tf.matmul(dict_calib[cfg._SET_CALIB[1]],dict_calib[cfg._SET_CALIB[2]])
    points3D = tf_coord_transform(points,trans_mat)
    pointsDist = points3D[:,2]

    # Project to image space
    trans_mat = tf.matmul(dict_calib[cfg._SET_CALIB[0]],trans_mat)
    points2D = tf_coord_transform(points,trans_mat)

    # Find only feasible points
    idx1 = (points2D[:,0]>=0) & (points2D[:,0] <=im_width-1)
    idx2 = (points2D[:,1]>=0) & (points2D[:,1] <=im_height-1)
    idx3 = (pointsDist>=0)
    idx_in = idx1 & idx2 & idx3
    points2D_fin = tf.boolean_mask(points2D,idx_in)
    pointsDist_fin = tf.boolean_mask(pointsDist,idx_in)

    return points2D_fin, pointsDist_fin

def tf_dist_to_pixel(val_dist, mode='inverse', d_max=_D_MAX, d_min=_D_MIN):
    """ Returns pixel value from distance measurment
    Args:
        val_dist: distance value (m)
        mode: 'inverse' vs 'standard'
        d_max: maximum distance to consider
        d_min: minimum distance to consider
    Returns:
        pixel value in 'uint8' format
    """
    val_dist = d_max if val_dist>d_max else val_dist if val_dist>d_min else d_min
    if mode == 'standard':
        return tf.cast(tf.round(val_dist*255.0/d_max),tf.uint8)
    elif mode == 'inverse':
        return tf.cast(tf.round(d_min*255.0/val_dist),tf.uint8)
    else:
        # Default is inverse
        return tf.cast(tf.round(d_min*255.0/val_dist),tf.uint8)

def tf_points_to_img(points2D,pointsDist,im_height,im_width):
    pointsPixel = tf_dist_to_pixel(pointsDist)
    return tf.scatter_nd(tf.cast(tf.round(points2D),tf.int32),pointsPixel,
                         [im_height,im_width])

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
            'lidar/points': tf.FixedLenFeature(
                (), dtype=tf.float32),
            'calib/mat_intrinsic': tf.FixedLenFeature(
                [3,4], tf.float32, default_value=tf.zeros([4,4], dtype=tf.float32)),
            'calib/mat_rect': tf.FixedLenFeature(
                [4,4], tf.float32, default_value=tf.zeros([4,4], dtype=tf.float32)),
            'calib/mat_extrinsic': tf.FixedLenFeature(
                [4,4], tf.float32, default_value=tf.zeros([3,4], dtype=tf.float32))
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
                [3], tf.float32, default_value=tf.zeros([3], dtype=tf.float32))
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
            'a_vec': slim.tfexample_decoder.Tensor('param/a_vec')
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
