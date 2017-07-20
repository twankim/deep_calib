# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-07 21:15:23
# @Last Modified by:   twankim
# @Last Modified time: 2017-07-20 14:17:40

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

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
    q_r[0] = np.cos(param_decalib['rot']*np.pi/360.0)
    q_r[1:] = np.sin(param_decalib['rot']*np.pi/360.0)*param_decalib['a_vec']
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