# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-06-21 11:55:51
# @Last Modified by:   twankim
# @Last Modified time: 2017-07-06 10:28:31

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

import _init_paths
from datasets.config import cfg

def main(args):
    if not args.is_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gid
    path_tf = args.path_tf
    f_train = args.f_train
    
    path_train = os.path.join(path_tf,f_train+'.tfrecord')
    path_cp = os.path.join(path_tf,f_train,'checkpoints')
    if not os.path.exists(path_cp):
        os.makedirs(path_cp)


        


def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1')

    parser = argparse.ArgumentParser(description=
                        'Calibration with deep learning')
    parser.add_argument('-gid', dest='gid',
                        help='CUDA_VISIBLE_DEVICES (Check your machine ID and ex) 0,1',
                        default = '0', type = str)
    parser.add_argument('-is_cpu', dest='is_cpu',
                        help='Use CPU only. True/False',
                        default = False, type = str2bool)
    parser.add_argument('-data', dest='path_tf',
                        help='Path to kitti TfRecord',
                        default = '/data/tf/kitti_calib',
                        type = str)
    parser.add_argument('-f_train', dest='f_train',
                        help='Name of TfRecord/Checkpoint (excluding .tfrecord)',
                        default = 'kitti_calib_20_1.5_train', type = str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print "Called with args:"
    print args
    sys.exit(main(args))