# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-07 21:15:23
# @Last Modified by:   twankim
# @Last Modified time: 2017-09-27 21:14:23

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage.io import imsave
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.slim as slim

from datasets.config import cfg

_D_MAX = 50.0
_D_MIN = 1.5

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
            'lidar/points': float_feature(points),
            'calib/mat_intrinsic': float_feature(mat_intrinsic),
            'calib/mat_rect': float_feature(mat_rect),
            'calib/mat_extrinsic': float_feature(mat_extrinsic)
            }))

def calib_to_tfexample_test(im_data, im_data_depth, im_format, height, width,
                            y_true, rot, a_vec, params_crop):
    return tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': bytes_feature(im_data),
            'image/format': bytes_feature(im_format),
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'lidar/encoded': bytes_feature(im_data_depth),
            'param/y_calib': float_feature(y_true),
            'param/rot_angle': float_feature(rot),
            'param/a_vec': float_feature(a_vec),
            'param/params_crop': int64_feature(params_crop)
            }))

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
        return np.round(minmax_scale(val_dist,
                                     d_min,d_max,
                                     0,255)).astype('uint8')
    elif mode == 'inverse':
        return np.round(minmax_scale(1.0/val_dist,
                                     1.0/d_max,1.0/d_min,
                                     0,255)).astype('uint8')
    else:
        # Default is inverse
        return np.round(minmax_scale(1.0/val_dist,
                                     1.0/d_max,1.0/d_min,
                                     0,255)).astype('uint8')

def points_to_img(points2D,pointsDist,im_height,im_width):
    print('!!!!min {}, max {}'.format(max(pointsDist),min(pointsDist)))
    points2D = np.round(points2D).astype('int')
    im_depth = np.zeros((im_height,im_width),dtype=np.uint8)
    for i in xrange(np.shape(points2D)[0]):
        x,y = points2D[i,:]
        im_depth[y,x] = dist_to_pixel(pointsDist[i])
        # im_depth[y,x] = dist_to_pixel(pointsDist[i],mode='standard')

    # Find LIDAR sensed region
    yx_max = np.max(points2D,axis=0)
    yx_min = np.min(points2D,axis=0)

    offset_height = yx_min[1]
    offset_width = yx_min[0]
    crop_height = yx_max[1]-yx_min[1]
    crop_width = yx_max[0]-yx_min[0]

    return [im_depth.reshape(im_height,im_width,1),
            [offset_height,offset_width,crop_height,crop_width]]


def tf_coord_transform(points, t_mat):
    # Change to homogeneous form
    points = tf.concat([points,tf.ones([tf.shape(points)[0],1],tf.float32)], 1)
    t_points = tf.matmul(points,tf.transpose(t_mat))
    # Normalize
    t_points = tf.div(t_points[:,:-1],tf.expand_dims(t_points[:,-1],1))
    return t_points

def tf_project_lidar_to_img(dict_calib,points,im_height,im_width):
    # Extract depth data first before projection to 2d image space
    trans_mat = tf.matmul(dict_calib[cfg._SET_CALIB[1]],
                          dict_calib[cfg._SET_CALIB[2]])
    points3D = tf_coord_transform(points,trans_mat)
    pointsDist = points3D[:,2]

    # Project to image space
    trans_mat = tf.matmul(dict_calib[cfg._SET_CALIB[0]],trans_mat)
    points2D = tf_coord_transform(points,trans_mat)

    # Find only feasible points
    idx1 = (points2D[:,0]>=0) & (points2D[:,0] <=tf.to_float(im_width)-1)
    idx2 = (points2D[:,1]>=0) & (points2D[:,1] <=tf.to_float(im_height)-1)
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
    val_dist = tf.maximum(val_dist,d_min)
    val_dist = tf.minimum(val_dist,d_max)
    if mode == 'standard':
        return tf.cast(tf.round(minmax_scale(val_dist,
                                             d_min,d_max,
                                             0,255)),tf.uint8)
    elif mode == 'inverse':
        return tf.cast(tf.round(minmax_scale(1.0/val_dist,
                                             1.0/d_max,1.0/d_min,
                                             0,255)),tf.uint8)
    else:
        # Default is inverse
        return tf.cast(tf.round(minmax_scale(1.0/val_dist,
                                             1.0/d_max,1.0/d_min,
                                             0,255)),tf.uint8)

def tf_points_to_img(points2D,pointsDist,im_height,im_width):
    pointsPixel = tf_dist_to_pixel(pointsDist)
    points2D_yx = tf.cast(tf.round(tf.reverse(points2D,axis=[1])),tf.int32)
    img = tf.scatter_nd(points2D_yx,pointsPixel,[im_height,im_width])

    # Find LIDAR sensed region
    yx_max = tf.reduce_max(points2D_yx,axis=0)
    yx_min = tf.reduce_min(points2D_yx,axis=0)

    offset_height = yx_min[0]
    offset_width = yx_min[1]
    crop_height = yx_max[0]-yx_min[0]
    crop_width = yx_max[1]-yx_min[1]

    return [tf.expand_dims(img, 2), 
            [offset_height,offset_width,crop_height,crop_width]]

def _crop(image, offset_height, offset_width, crop_height, crop_width):
    """Crops the given image using the provided offsets and sizes.

    Note that the method doesn't assume we know the input image size but it does
    assume we know the input image rank.

    Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

    Returns:
    the cropped (and resized) image.

    Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
        less than the crop size.
    """
    original_shape = tf.shape(image)

    rank_assertion = tf.Assert(
        tf.equal(tf.rank(image), 3),
        ['Rank of image must be equal to 3.'])
    with tf.control_dependencies([rank_assertion]):
        cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

    size_assertion = tf.Assert(
        tf.logical_and(
            tf.greater_equal(original_shape[0], crop_height),
            tf.greater_equal(original_shape[1], crop_width)),
        ['Crop size greater than the image size.'])

    offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

    # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
    # define the crop size.
    with tf.control_dependencies([size_assertion]):
        image = tf.slice(image, offsets, cropped_shape)
    return tf.reshape(image, cropped_shape)

def tf_crop_lidar_image(image,lidar,params_crop):
    """Crop RGB image and LIDAR image to consider only 
        LIDAR-sensed region
    Return:
        cropped image and lidar
    """
    params_crop = tf.cast(params_crop,tf.int32)
    offset_height = params_crop[0]
    offset_width = params_crop[1]
    crop_height = params_crop[2]
    crop_width = params_crop[3]
    image = _crop(image,offset_height,offset_width,crop_height,crop_width)
    lidar = _crop(lidar,offset_height,offset_width,crop_height,crop_width)

    image.set_shape([None, None, 3])
    lidar.set_shape([None, None, 1])

    return image, lidar

def tf_prepare_train(image,points,
                     mat_intrinsic,mat_rect,mat_extrinsic,
                     max_theta,max_dist):
    # Prepare image and lidar image for training
    im_shape = tf.shape(image)
    im_height = im_shape[0]
    im_width = im_shape[1]

    param_rands = gen_ran_decalib(max_theta,max_dist,1)

    param_decalib = gen_decalib(max_theta,max_dist,param_rands,0)
    y_true = tf.constant(param_decalib['y'],dtype=tf.float32)

    # Intrinsic parameters and rotation matrix (for reference cam)
    ran_dict = {}
    ran_dict[cfg._SET_CALIB[0]] = mat_intrinsic
    ran_dict[cfg._SET_CALIB[1]] = mat_rect
    # Extrinsic parameters to decalibrated ones
    ran_dict[cfg._SET_CALIB[2]] = tf.matmul(
           mat_extrinsic,
           tf.constant(quat_to_transmat(param_decalib['q_r'],
                                        param_decalib['t_vec']),
                       dtype=tf.float32))

    points2D_ran, pointsDist_ran = tf_project_lidar_to_img(ran_dict,
                                                           points,
                                                           im_height,
                                                           im_width)
    lidar,params_crop = tf_points_to_img(
                        points2D_ran,pointsDist_ran,im_height,im_width)
    image,lidar = tf_crop_lidar_image(image,lidar,params_crop)

    return image,lidar,y_true

def tf_prepare_test(image,lidar,params_crop):
    return tf_crop_lidar_image(image,lidar,params_crop)

def imlidarwrite(fname,im,im_depth):
    """Write image with RGB and depth
    Args:
        fname: file name
        im: RGB image array (h x w x 3)
        im_depth: depth image array (h x w)
    """
    im_out = im.copy()
    im_depth = np.squeeze(im_depth,axis=2)
    idx_h, idx_w = np.nonzero(im_depth)
    cmap = plt.get_cmap('jet')
    for i in xrange(len(idx_h)):
        im_out[idx_h[i],idx_w[i],:] = (255*np.array(
                        cmap(im_depth[idx_h[i],idx_w[i]]/255.0)[:3]))\
                        .astype(np.uint8)
    imsave(fname,im_out)
    print("!!! Write:{}".format(fname))