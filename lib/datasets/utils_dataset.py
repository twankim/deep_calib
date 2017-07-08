# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-07 21:15:23
# @Last Modified by:   twankim
# @Last Modified time: 2017-07-07 21:19:32

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

# theta (degree) (randomly generate unit vector for axis, and rotate theta)
# dist (m) for (x,y,z)
def gen_decalib(max_theta, max_dist):
    assert (max_theta>0) & (max_dist>0),\
            "Maximum values of angle (degree) and distance (m) must be positive"
    assert max_theta<=30,\
            "Support angle from -30 ~ 30"
    param_decalib = {}

    # Rotation angle
    param_decalib['rot'] = np.random.uniform(-max_theta,max_theta)

    # Rotation axis (unit vector)
    a_vec = np.random.standard_normal(3)
    param_decalib['a_vec'] = a_vec/np.linalg.norm(a_vec)

    # Translation vector
    dist = np.random.uniform(0,max_dist)
    unit_vec = np.random.standard_normal(3)
    param_decalib['trans'] = dist*unit_vec/np.linalg.norm(unit_vec)

    # Dual quaternions
    q_r = np.zeros(4)
    q_r[0] = np.cos(param_decalib['rot']*np.pi/90.0)
    q_r[1:] = np.sin(param_decalib['rot']*np.pi/90.0)*param_decalib['a_vec']
    param_decalib['q_r'] = q_r

    q_t = np.zeros(4)
    q_t[1:] = param_decalib['trans']
    param_decalib['q_t'] = q_t

    param_decalib['y'] = np.concatenate((q_r,q_t[1:]))

    return param_decalib

# Fuctions from TF-Slim dataset_utils
def int64_feature(values):
    """Returns a TF-Feature of int64s.
    Args:
        values: A scalar or list of values.
    Returns:
        a TF-Feature.
    """
    if isinstance(values,np.ndarray):
        values = list(values)
    elif not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def float_feature(values):
    """Returns a TF-Feature of floats.
    Args:
        values: A scalar or list of values.
    Returns:
        a TF-Feature.
    """
    if isinstance(values,np.ndarray):
        values = list(values)
    elif not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.
    Args:
        values: A string.
    Returns:
        a TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def calib_to_tfexample(im_data, im_data_depth, im_format, height, width,
                       y_true, rot, a_vec):
    return tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(im_data),
            'image/format': bytes_feature(im_format),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'lidar/encoded': bytes_feature(im_data_depth),
            'param/y_calib': float_feature(y_true),
            'param/rot_angle': float_feature(rot),
            'param/a_vec': float_feature(a_vec)
            }))