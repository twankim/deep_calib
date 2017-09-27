# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-07 21:15:23
# @Last Modified by:   twankim
# @Last Modified time: 2017-09-27 01:44:53

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage.io import imsave
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
    Rt[0,1] = 2*(x*y - w*z)
    Rt[0,2] = 2*(x*z + w*y)
    Rt[1,0] = 2*(x*y + w*z)
    Rt[1,1] = w**2 - x**2 + y**2 - z**2
    Rt[1,2] = 2*(y*z - w*x)
    Rt[2,0] = 2*(x*z - w*y)
    Rt[2,1] = 2*(y*z + w*x)
    Rt[2,2] = w**2 - x**2 - y**2 + z**2

    Rt[:3,3] = t_vec
    return Rt

def minmax_scale(x,i_min,i_max,o_min,o_max):
    # MinMax scaling of x
    # i_min<= x <= i_max to o_min<= x_new <= o_max
    return (x-i_min)/float(i_max-i_min)*(o_max-o_min)+o_min

def qr_to_yr(q_r,max_theta):
    # Normalize q_r (rotation quaternion) to [-1,1]
    assert (max_theta>0) & (max_theta<180),\
            "Maximum value of max_theta(degree) must be in (0,180)"
    y_r = np.zeros(4)
    y_r[0] = minmax_scale(q_r[0],np.cos(max_theta*np.pi/360.0),1,-1,1)
    y_r[1] = minmax_scale(q_r[1],
                          np.sin(max_theta*np.pi/360.0),
                         -np.sin(max_theta*np.pi/360.0),
                         -1,1)
    y_r[2] = minmax_scale(q_r[2],
                          np.sin(max_theta*np.pi/360.0),
                         -np.sin(max_theta*np.pi/360.0),
                         -1,1)
    y_r[3] = minmax_scale(q_r[3],
                          np.sin(max_theta*np.pi/360.0),
                         -np.sin(max_theta*np.pi/360.0),
                         -1,1)
    return y_r

def yr_to_qr(y_r,max_theta):
    # Scale y_r back to q_r(rotation quaternion)
    assert (max_theta>0) & (max_theta<180),\
            "Maximum value of max_theta(degree) must be in (0,180)"
    q_r = np.zeros(4)
    q_r[0] = minmax_scale(y_r[0],-1,1,np.cos(max_theta*np.pi/360.0),1)
    q_r[1] = minmax_scale(y_r[1],-1,1,
                          np.sin(max_theta*np.pi/360.0),
                          -np.sin(max_theta*np.pi/360.0))
    q_r[2] = minmax_scale(y_r[2],-1,1,
                          np.sin(max_theta*np.pi/360.0),
                          -np.sin(max_theta*np.pi/360.0))
    q_r[3] = minmax_scale(y_r[3],-1,1,
                          np.sin(max_theta*np.pi/360.0),
                          -np.sin(max_theta*np.pi/360.0))
    return q_r

def gen_ran_decalib(max_theta, max_dist, num_gen):
    param_rands = {}
    # Rotation angle
    param_rands['rot'] = np.random.uniform(-max_theta,max_theta,num_gen)
    
    # Rotation axis (unit vectors)
    a_vecs = np.random.standard_normal((3,num_gen))
    a_vecs = (a_vecs/np.linalg.norm(a_vecs,axis=0)).T
    param_rands['a_vec'] = [a_vecs[i] for i in xrange(num_gen)]

    # Translation vector
    dists = np.random.uniform(0,max_dist,num_gen)
    t_vecs = np.random.standard_normal((3,num_gen))
    t_vecs = (dists*(t_vecs/np.linalg.norm(t_vecs,axis=0))).T
    param_rands['t_vec'] = [t_vecs[i] for i in xrange(num_gen)]

    return param_rands
    

def gen_decalib(max_theta, max_dist, param_rands, i_ran):
    assert (max_theta>0) & (max_dist>0),\
            "Maximum values of angle (degree) and distance (m) must be positive"
    assert max_theta<=30,\
            "Support angle from -30 ~ 30"
    param_decalib = {}

    # Rotation angle
    param_decalib['rot'] = param_rands['rot'][i_ran]

    # Rotation axis (unit vector)
    param_decalib['a_vec'] = param_rands['a_vec'][i_ran]

    # Translation vector
    param_decalib['t_vec'] = param_rands['t_vec'][i_ran]

    # Quaternion for rotation
    q_r = np.zeros(4)
    q_r[0] = np.cos(param_decalib['rot']*np.pi/360.0)
    q_r[1:] = np.sin(param_decalib['rot']*np.pi/360.0)*param_decalib['a_vec']
    param_decalib['q_r'] = q_r

    param_decalib['y'] = np.concatenate((qr_to_yr(q_r,max_theta),
                                         param_decalib['t_vec']))

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
        values = values.tolist()
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

def calib_to_tfexample_train(im_data, im_format, height, width,
                             points, mat_intrinsic, mat_rect, mat_extrinsic):
    return tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(im_data),
            'image/format': bytes_feature(im_format),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'lidar/points': bytes_feature(points),
            'calib/mat_intrinsic': bytes_feature(mat_intrinsic),
            'calib/mat_rect': bytes_feature(mat_rect),
            'calib/mat_extrinsic': bytes_feature(mat_extrinsic)
            }))

def calib_to_tfexample_test(im_data, im_data_depth, im_format, height, width,
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
        im_depth: depth image array (h x w)
    """
    im_out = im.copy()
    idx_h, idx_w = np.nonzero(im_depth)
    cmap = plt.get_cmap('jet')
    for i in xrange(len(idx_h)):
        im_out[idx_h[i],idx_w[i],:] = (255*np.array(
                        cmap(im_depth[idx_h[i],idx_w[i]]/255.0)[:3]))\
                        .astype(np.uint8)
    imsave(fname,im_out)
    print("!!! Write:{}".format(fname))