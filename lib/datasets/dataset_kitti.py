# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-05 13:32:38
# @Last Modified by:   twankim
# @Last Modified time: 2017-07-07 21:20:01

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets.config import cfg

_NAME_DATA = 'kitti'

_NUM_TRAIN = 7481
_NUM_TEST = 7518

_NUM_SAMPLES = {'train': _NUM_TRAIN*cfg._NUM_GEN,
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

def coord_transform(points, t_mat):
    # Change to homogeneous form
    points = np.hstack([points,np.ones((np.shape(points)[0],1))])
    t_points = np.dot(points,t_mat.T)
    # Normalize
    t_points = t_points[:,:-1]/t_points[:,[-1]]
    return t_points

def project_velo_to_img(dict_calib,points,im_height,im_width):
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

def dist_to_pixel(val_dist, mode='inverse', d_max=100, d_min =1):
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
        return np.round(255.0/val_dist).astype('uint8')
    else:
        # Default is inverse
        return np.round(255.0/val_dist).astype('uint8')

def points_to_img(points2D,pointsDist,im_height,im_width):
    im_depth = np.zeros((im_height,im_width),dtype=np.uint8)
    for i in xrange(np.shape(points2D)[0]):
        x,y = np.round(points2D[i,:]).astype('int')
        im_depth[y,x] = dist_to_pixel(pointsDist[i])
    return im_depth

# Product of quaternions
def qprod(q_a,q_b):
    assert np.shape(q_a) == (4,),\
            "!! Size of q_a should be 4"
    assert np.shape(q_b) == (4,),\
            "!! Size of q_b should be 4"
    
    out = np.zeros(np.shape(q_a))
    out[0] = q_a[0]*q_b[0] - np.dot(q_a[1:],q_b[1:])
    out[1:] = q_a[0]*q_b[1:] + q_b[0]*q_a[1:] + np.cross(q_a[1:],q_b[1:])
    return out

# Dual quaternion to 4x4 homogenous transform matrix
# q = q_r + 0.5eps q_t q_r
def dualquat_to_transmat(q_r,q_t):
    assert np.shape(q_r) == (4,),\
            "!! Size of q_r should be 4"
    assert np.shape(q_t) == (4,),\
            "!! Size of q_t should be 4"
    # assert np.linalg.norm(q_r) == 1,\
    #         "q_r must be normalized. (||q_r|| = 1)"
    assert q_t[0] == 0,\
            "!! Real part of q_t must be 1"

    Rt = np.zeros((4,4))
    Rt[3,3] = 1
    w,x,y,z = q_r

    Rt[0,0] = w**2 + x**2 - y**2 - z**2
    Rt[0,1] = 2*x*y - 2*w*z
    Rt[0,2] = 2*x*z + 2*w*y
    Rt[1,0] = 2*x*y + 2*w*z
    Rt[1,1] = w**2 - x**2 + y**2 - z**2
    Rt[1,2] = 2*y*z - 2*w*x
    Rt[2,0] = 2*x*z - 2*w*y
    Rt[2,1] = 2*y*z + 2*w*x
    Rt[2,2] = w**2 - x**2 - y**2 + z**2

    # q_rc = q_r.copy()
    # q_rc[1:] = -q_rc[1:]
    # q_t = 2*qprod(q_d,q_rc)
    Rt[:3,3] = q_t[1:]
    return Rt

def get_data(path_data,image_set,reader=None):
    assert image_set in _NUM_SAMPLES,\
            "!! {} data is not supported".format(image_set)
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.VarLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='raw'),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'lidar/encoded': bytes_feature(im_data_depth),
        'param/y_calib': float_feature(y_true),
        'param/rot_angle': float_feature(rot),
        'param/a_vec': float_feature(a_vec)
    }

    items_to_handlers = {}

    decoder = slim.tfexample_decoder.TFExampleDecoder(
                    keys_to_features, items_to_handlers)

    return slim.dataset.Dataset(
            data_sources=path_data,
            reader=reader,
            decoder=decoder,
            num_samples=_NUM_SAMPLES[image_set],
            items_to_descriptions=None)
