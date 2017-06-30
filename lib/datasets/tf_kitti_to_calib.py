# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-06-26 16:55:00
# @Last Modified by:   twankim
# @Last Modified time: 2017-06-30 14:00:49

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import glob

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

_TASK = 'object'

_TYPE_CALIB = 'calib'
_TYPE_IMAGE = 'image_2'
_TYPE_VELO = 'velodyne'

_DIR_CALIB = 'data_{}_{}'.format(_TASK,_TYPE_CALIB)
_DIR_IMAGE = 'data_{}_{}'.format(_TASK,_TYPE_IMAGE)
_DIR_VELO = 'data_{}_{}'.format(_TASK,_TYPE_VELO)

_SET_CALIB = ['P2','R0_rect','Tr_velo_to_cam']

def conv_calib(vid,cal_val):
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
    param_decalib['rot'] = np.random.uniform(-max_theata,max_theta)

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

    return param_decalib

def main(args):
    path_kitti = args.path_kitti
    assert os.path.exists(path_kitti),\
            "Download KITTI Dataset or enter correct path"
    
    path_out = args.path_out
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    path_cal = os.path.join(path_kitti,_TASK,_DIR_CALIB)
    path_velo = os.path.join(path_kitti,_TASK)

    for image_set in ['training','testing']:
        imList = glob.glob(os.path.join(path_cal,image_set,_TYPE_CALIB,'*.txt'))
        imList.sort()
        imNames = [os.path.split(d)[1].strip('.txt') for d in imList]

        for iter, imName in enumearate(imNames):
            f_calib = os.path.join(path_cal,image_set,_TYPE_CALIB,imName+'.txt')
            # Read & Save calibration matrix
            with open(f_calib,'r'):
                temp_dict = {}
                for line in input_file:
                    if len(line) > 1:
                        vid,vals = line.split(':',1)
                        val_cal = [float(v) for v in vals.split()]
                        if vid in _SET_CALIB:
                            temp_dict[vid] = conv_calib(vid,np.array(val_cal))

            # Read velodyne points
            f_velo = os.path.join()


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
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print "Called with args:"
    print args
    sys.exit(main(args))