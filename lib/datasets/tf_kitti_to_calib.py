# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-06-26 16:55:00
# @Last Modified by:   twankim
# @Last Modified time: 2017-06-30 15:38:48

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import glob
import cv2

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

_TASK = 'object'

_TYPE_CALIB = 'calib'
_TYPE_IMAGE = 'image_2'
_TYPE_VELO = 'velodyne'

_FORMAT_CALIB = '.txt'
_FORMAT_IMAGE = '.png'
_FORMAT_VELO = '.bin'

_DIR_CALIB = 'data_{}_{}'.format(_TASK,_TYPE_CALIB)
_DIR_IMAGE = 'data_{}_{}'.format(_TASK,_TYPE_IMAGE)
_DIR_VELO = 'data_{}_{}'.format(_TASK,_TYPE_VELO)

_NUM_GEN = 5 # Number of random generation per image

_SET_CALIB = ['P2','R0_rect','Tr_velo_to_cam']

def get_calib_mat(vid,cal_val):
    assert vid in cal_set, 'Wrong parsing occurred: {}'.format(vid)
    if vid == cal_set[0]: # P2
        return cal_val.reshape((3,4))
    elif vid == cal_set[1]: # R0_rect
        cal_mat = np.zeros((4,4))
        cal_mat[:3,:3] = cal_val.reshape((3,3))
        cal_mat[3,3] = 1
        return cal_mat
    else: # Tr_velo_to_cam
        cal_mat = np.zeros((4,4))
        cal_mat[:3,:4] = cal_val.reshape((3,4))
        cal_mat[3,3] = 1
        return cal_mat

def coord_transform(points, t_mat):
    # Change to homogeneous form
    points = np.hstack([points,np.ones((np.shape(points)[0],1))])
    t_points = np.dot(points,t_mat.T)
    # Normalize
    t_points = t_points[:,:-1]/t_points[:,[-1]]
    return t_points

def project_velo_to_img(dict_calib,points,im_height,im_width):
    # Extract depth data first before projection to 2d image space
    trans_mat = np.dot(dict_calib[_SET_CALIB[1]],dict_calib[_SET_CALIB[2]])
    points3D = coord_transform(points,trans_mat)
    pointsDist = points3D[:,2]

    # Project to image space
    trans_mat = np.dot(dict_calib[_SET_CALIB[0]],trans_mat)
    points2D = coord_transform(points,trans_mat)

    # Find only feasible points
    idx1 = (points2D[:,0]>=0) & (points2D[:,0] <=im_width-1)
    idx2 = (points2D[:,1]>=0) & (points2D[:,1] <=im_height-1)
    idx3 = (pointsDist>=0)
    idx_in = idx1 & idx2 & idx3
    points2D_fin = points2D[idx_in,:]
    pointsDist_fin = pointsDist[idx_in]

    return points2D_fin, pointsDist_fin

# Product of quaternions
def qprod(q_a,q_b):
    assert np.shape(q_a) == 4,\
            "Size of q_a should be 4"
    assert np.shape(q_b) == 4,\
            "Size of q_b should be 4"
    
    out = np.zeros(np.shape(q_a))
    out[0] = q_a[0]*q_b[0] - np.dot(q_a[1:],q_b[1:])
    out[1:] = q_a[0]*q_b[1:] + q_b[0]*q_a[1:] + np.cross(q_a[1:],q_b[1:])
    return out

# Dual quaternion to 4x4 homogenous transform matrix
# q = q_r + 0.5eps q_t q_r
def dualquat_to_transmat(q_r,q_t):
    assert np.shape(q_r) == 4,\
            "Size of q_r should be 4"
    assert np.shape(q_t) == 4,\
            "Size of q_t should be 4"
    assert np.linalg.norm(q_r) == 1,\
            "q_r must be normalized. (||q_r|| = 1)"
    assert q_t[0] == 0,\
            "real part of q_t must be 1"

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
    q_r[0] = np.cos(param_decalib['rot']*np.pi/90.0)
    q_r[1:] = np.sin(param_decalib['rot']*np.pi/90.0)*param_decalib['a_vec']
    param_decalib['q_r'] = q_r

    q_t = np.zeros(4)
    q_t[1:] = param_decalib['trans']
    param_decalib['q_t'] = q_t

    param_decalib['y'] = np.concatenate((q_r,q_t[1:]))

    return param_decalib

def main(args):
    max_theta = args.max_theta
    max_dist = args.max_dist

    path_kitti = args.path_kitti
    assert os.path.exists(path_kitti),\
            "Download KITTI Dataset or enter correct path"
    
    path_out = args.path_out
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    path_cal = os.path.join(path_kitti,_TASK,_DIR_CALIB)
    path_velo = os.path.join(path_kitti,_TASK,_DIR_VELO)
    path_img = os.path.join(path_kitti,_TASK,_DIR_IMAGE)

    for image_set in ['training','testing']:
        imList = glob.glob(os.path.join(path_cal,image_set,_TYPE_CALIB,'*'+_FORMAT_CALIB))
        imList.sort()
        imNames = [os.path.split(d)[1].strip('.txt') for d in imList]

        num_data = len(imNames)*_NUM_GEN # Number of data to be gerated

        for iter, imName in enumearate(imNames):
            f_calib = os.path.join(path_cal,image_set,_TYPE_CALIB,imName+_FORMAT_CALIB)
            # Read & Save calibration matrix
            with open(f_calib,'r'):
                temp_dict = {}
                for line in input_file:
                    if len(line) > 1:
                        vid,vals = line.split(':',1)
                        val_cal = [float(v) for v in vals.split()]
                        if vid in _SET_CALIB:
                            temp_dict[vid] = get_calib_mat(vid,np.array(val_cal))

            # Read velodyne points
            f_velo = os.path.join(path_velo,image_set,_TYPE_VELO,imName+_FORMAT_VELO)
            points_org = np.fromfile(f_velo,dtype=np.float32).reshape(-1,4)
            points = points_org[:,:3] # exclude points reflectance

            # Read image file
            f_img = os.path.join(path_img,image_set,_TYPE_IMAGE,imName+_FORMAT_IMAGE)
            im = cv2.imread(f_img)
            im = im[:,:,(2,1,0)] # BGR to RGB
            im_height,im_width = np.shape(im)[0:2]

            # Project velodyne points to image plane
            points2D, pointsDist = project_velo_to_img(temp_dict,
                                                       points,
                                                       im_height,
                                                       im_width)

            # ------- Generate random ratation for decalibration data --------
            # Generate random rotation
            for i_ran in xrange(_NUM_GEN):
                param_decalib = gen_decalib(max_theta, max_dist)
                ran_dict = temp_dict.copy()
                ran_dict[_SET_CALIB[2]] = dualquat_to_transmat(param_decalib['q_r'],
                                                               param_decalib['q_t'])
            
                points2D_ran, pointsDist_ran = project_velo_to_img(ran_dict,
                                                                   points,
                                                                   im_height,
                                                                   im_width)


def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1')

    parser = argparse.ArgumentParser(description=
                        'Dataset conversion to TF format')
    parser.add_argument('-dir_in', dest='path_kitti',
                        help='Path to kitti dataset',
                        default = '/data/kitti', type = str)
    parser.add_argument('-dir_out', dest='path_out',
                        help='Path to save tfrecord kitti dataset',
                        default = '/data/tf/kitti_calib', type = str)
    parser.add_argument('-max_theta', dest=max_theta,
                        help='Range of rotation angle in degree [-theta,+theta)',
                        default = 10, type=float)
    parser.add_argument('-max_dist', dest=max_dist,
                        help='Maximum translation distance in meter',
                        default = 1.5, type=float)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print "Called with args:"
    print args
    sys.exit(main(args))