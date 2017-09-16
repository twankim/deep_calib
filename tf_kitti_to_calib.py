# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-06-26 16:55:00
# @Last Modified by:   twankim
# @Last Modified time: 2017-09-16 17:23:27

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

import _init_paths
from datasets.config import cfg
from datasets.dataset_kitti import (get_calib_mat,
                                    project_lidar_to_img,
                                    points_to_img)
                                    
from datasets.utils_dataset import *

def main(args):
    if not args.is_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gid
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    max_theta = args.max_theta
    max_dist = args.max_dist

    verbose = args.verbose

    path_kitti = args.path_kitti
    assert os.path.exists(path_kitti),\
            "Download KITTI Dataset or enter correct path"
    
    path_out = args.path_out
    if not os.path.exists(path_out):
        os.makedirs(path_out)

    path_cal = os.path.join(path_kitti,cfg._TASK,cfg._DIR_CALIB)
    path_velo = os.path.join(path_kitti,cfg._TASK,cfg._DIR_VELO)
    path_img = os.path.join(path_kitti,cfg._TASK,cfg._DIR_IMAGE)

    for image_set in ['training','testing']:
        f_tfrec = os.path.join(path_out,
                               cfg._TF_FORMAT.format(
                                        'kitti',
                                        max_theta,
                                        max_dist,
                                        image_set.split('ing')[0]))

        imList = glob.glob(os.path.join(path_cal,image_set,cfg._TYPE_CALIB,'*'+cfg._FORMAT_CALIB))
        imList.sort()
        imNames = [os.path.split(d)[1].strip('.txt') for d in imList]

        num_data = len(imNames)*cfg._NUM_GEN # Number of data to be gerated

        print("... Writing {} set".format(image_set))

        with tf.python_io.TFRecordWriter(f_tfrec) as tfrecord_writer:
            with tf.Graph().as_default():
                with tf.Session('') as sess:
                    for iter, imName in enumerate(imNames):
                        # Get original calibration info
                        f_calib = os.path.join(path_cal,image_set,cfg._TYPE_CALIB,imName+cfg._FORMAT_CALIB)
                        temp_dict = get_calib_mat(f_calib)

                        # Read velodyne points
                        f_velo = os.path.join(path_velo,image_set,cfg._TYPE_VELO,imName+cfg._FORMAT_VELO)
                        points_org = np.fromfile(f_velo,dtype=np.float32).reshape(-1,4)
                        points = points_org[:,:3] # exclude points reflectance

                        # Read image file
                        f_img = os.path.join(path_img,image_set,cfg._TYPE_IMAGE,imName+cfg._FORMAT_IMAGE)
                        im = cv2.imread(f_img)
                        im = im[:,:,(2,1,0)] # BGR to RGB
                        im_height,im_width = np.shape(im)[0:2]
        
                        # Project velodyne points to image plane
                        points2D, pointsDist = project_lidar_to_img(temp_dict,
                                                                   points,
                                                                   im_height,
                                                                   im_width)

                        # !!!! For debugging
                        # im_depth_ho = points_to_img(points2D,pointsDist,im_height,im_width)
                        # cv2.imwrite('data_ex/ho_{}.png'.format(image_set),im_depth_ho)

                        # ------- Generate random ratation for decalibration data --------
                        # Generate random rotation
                        for i_ran in xrange(cfg._NUM_GEN):
                            param_decalib = gen_decalib(max_theta, max_dist)
                            # Copy intrinsic parameters and rotation matrix for reference cam
                            ran_dict = temp_dict.copy()
                            # Replace extrinsic parameters to decalibrated ones
                            ran_dict[cfg._SET_CALIB[2]] = np.dot(
                                        ran_dict[cfg._SET_CALIB[2]],
                                        quat_to_transmat(param_decalib['q_r'],param_decalib['t_vec']))
                    
                            points2D_ran, pointsDist_ran = project_lidar_to_img(ran_dict,
                                                                               points,
                                                                               im_height,
                                                                               im_width)
                            im_depth = points_to_img(points2D_ran,
                                                     pointsDist_ran,
                                                     im_height,
                                                     im_width)

                            # !!!! For debugging
                            # cv2.imwrite('data_ex/ho_{}_{}.png'.format(image_set,i_ran),im_depth)
                            # print('  - Angle:{}, nonzero:{}'.format(
                            #                         param_decalib['rot'],
                            #                         sum(sum(im_depth>0)))                                                    )

                            im_placeholder = tf.placeholder(dtype=tf.uint8)
                            im_depth_placeholder = tf.placeholder(dtype=tf.uint8)
                            encoded_image = tf.image.encode_png(im_placeholder)
                            encoded_image_depth = tf.image.encode_png(im_depth_placeholder)

                            if verbose:
                                sys.stdout.write('... ({}) Writing file to TfRecord {}/{}\n'.format(
                                                    image_set,cfg._NUM_GEN*iter+i_ran+1,num_data))
                                sys.stdout.flush()

                            png_string = sess.run(encoded_image,
                                                  feed_dict={im_placeholder:im})
                            png_string_depth = sess.run(encoded_image_depth,
                                                  feed_dict={im_depth_placeholder:im_depth.\
                                                                reshape(im_height,im_width,1)})

                            example = calib_to_tfexample(png_string,
                                                         png_string_depth,
                                                         b'png',
                                                         im_height,
                                                         im_width,
                                                         param_decalib['y'],
                                                         param_decalib['rot'],
                                                         param_decalib['a_vec']
                                                         )
                            tfrecord_writer.write(example.SerializeToString())

def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1')

    parser = argparse.ArgumentParser(description=
                        'Dataset conversion to TF format')
    parser.add_argument('-gid', dest='gid',
                        help='CUDA_VISIBLE_DEVICES (Check your machine ID and ex) 0,1',
                        default = '0', type = str)
    parser.add_argument('-is_cpu', dest='is_cpu',
                        help='Use CPU only. True/False',
                        default = False, type = str2bool)
    parser.add_argument('-dir_in', dest='path_kitti',
                        help='Path to kitti dataset',
                        default = '/data/kitti', type = str)
    parser.add_argument('-dir_out', dest='path_out',
                        help='Path to save tfrecord kitti dataset',
                        default = '/data/tf/kitti_calib', type = str)
    parser.add_argument('-max_theta', dest='max_theta',
                        help='Range of rotation angle in degree [-theta,+theta)',
                        default = 20, type=int)
    parser.add_argument('-max_dist', dest='max_dist',
                        help='Maximum translation distance in meter',
                        default = 1.5, type=float)
    parser.add_argument('-verbose', dest='verbose',
                        help='True: Print every data, False: print only train/test',
                        default = False, type=bool)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print ("Called with args:")
    print (args)
    sys.exit(main(args))