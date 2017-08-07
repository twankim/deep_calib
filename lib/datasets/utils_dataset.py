# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-07 21:15:23
# @Last Modified by:   twankim
# @Last Modified time: 2017-08-07 18:14:57

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import matplotlib.pyplot as plt

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

# Conjugate of a quaternion
def qconj(q_a):
    assert np.shape(q_a) == (4,),\
            "!! Size of q_a should be 4"

    out = q_a.copy()
    out[1:] = -out[1:]
    return out

def quat_to_transmat(q_r,t_vec):
    """ 
    Quaternion & Translation to 4x4 homogenous transform matrix
        q_r: unit quaternion for rotation
        t_vec: 3D translation vector
    * Note: Used quaternion istead of dual quaternion for simplicity
            in parameter learning
    """
    assert np.shape(q_r) == (4,),\
            "!! Size of q_r should be 4"
    assert np.shape(t_vec) == (3,),\
            "!! Size of t_vec should be 3"
    
    # Normalize quaternion (Just in case)
    if np.linalg.norm(q_r) != 1:
        q_r = q_r/np.linalg.norm(q_r)

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

    Rt[:3,3] = t_vec
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
    param_decalib['t_vec'] = dist*unit_vec/np.linalg.norm(unit_vec)

    # Quaternion for rotation
    q_r = np.zeros(4)
    q_r[0] = np.cos(param_decalib['rot']*np.pi/360.0)
    q_r[1:] = np.sin(param_decalib['rot']*np.pi/360.0)*param_decalib['a_vec']
    param_decalib['q_r'] = q_r

    param_decalib['y'] = np.concatenate((q_r,param_decalib['t_vec']))

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

def imlidarwrite(fname,im,im_depth):
    """Write image with RGB and depth
    Args:
        fname: file name
        im: RGB image array (h x w x 3)
        im_depth: depth image array (h x w x 1)
    """
    im_out = im.copy()
    idx_h, idx_w = np.nonzero(im_out)
    cmap = plt.get_cmap('jet')
    for i in xrange(len(idx_h)):
        im_out[idx_h[i],idx_w[i]] = (255*np.array(
                        cmap(im_depth[idx_h[i],idx_w[i]]/255.0)[:3]))\
                        .astype(np.uint8)
    cv2.imwrite(fname,im_out)