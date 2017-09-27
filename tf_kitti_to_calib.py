# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-06-26 16:55:00
# @Last Modified by:   twankim
# @Last Modified time: 2017-09-27 10:47:25

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import glob
from skimage.io import (imread,imsave)
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

    if args.is_train:
        image_set = 'training'
        f_tfrec = os.path.join(path_out,cfg._TF_FORMAT_TRAIN.format(
                                            'kitti',
                                            image_set.split('ing')[0]))
    else:
        max_theta = args.max_theta
        max_dist = args.max_dist
        image_set = 'testing'
        f_tfrec = os.path.join(path_out,cfg._TF_FORMAT_TEST.format(
                                            'kitti',
                                            image_set.split('ing')[0],
                                            max_theta,
                                            max_dist))

    imList = glob.glob(os.path.join(path_cal,
                                    image_set,
                                    cfg._TYPE_CALIB,
                                    '*'+cfg._FORMAT_CALIB))
    imList.sort()
    imNames = [os.path.split(d)[1].strip('.txt') for d in imList]

    print("... Writing {} set".format(image_set))

    with tf.python_io.TFRecordWriter(f_tfrec) as tfrecord_writer:
        with tf.Graph().as_default():
            with tf.Session('') as sess:
                for iter, imName in enumerate(imNames):
                    # Get original calibration info
                    f_calib = os.path.join(path_cal,
                                           image_set,
                                           cfg._TYPE_CALIB,
                                           imName+cfg._FORMAT_CALIB)
                    calib_dict = get_calib_mat(f_calib)

                    # Read velodyne points
                    f_velo = os.path.join(path_velo,
                                          image_set,
                                          cfg._TYPE_VELO,
                                          imName+cfg._FORMAT_VELO)
                    points_org = np.fromfile(f_velo,dtype=np.float32)
                    # exclude points reflectance
                    points = points_org.reshape(-1,4)[:,:3]

                    # Read image file
                    f_img = os.path.join(path_img,
                                         image_set,
                                         cfg._TYPE_IMAGE,
                                         imName+cfg._FORMAT_IMAGE)
                    im = imread(f_img)
                    im_height,im_width = np.shape(im)[0:2]

                    # For training, generate random decalib while training,
                    # For testing, generate fixed random decalib.
                    if image_set == 'training':
                        if verbose:
                            sys.stdout.write(
                                '... ({}) Writing file to TfRecord {}/{}\n'.format(
                                                    image_set,iter+1,len(imNames)))
                            sys.stdout.flush()
                        # Write to tfrecord
                        im_placeholder = tf.placeholder(dtype=tf.uint8)
                        encoded_image = tf.image.encode_png(im_placeholder)
                        png_string = sess.run(encoded_image,
                                              feed_dict={im_placeholder:im})
                        mat_intrinsic = calib_dict[cfg._SET_CALIB[0]].flatten()
                        mat_rect = calib_dict[cfg._SET_CALIB[1]].flatten()
                        mat_extrinsic = calib_dict[cfg._SET_CALIB[2]].flatten()
                        example = calib_to_tfexample_train(
                                                    png_string,
                                                    b'png',
                                                    im_height,
                                                    im_width,
                                                    points_org,
                                                    mat_intrinsic,
                                                    mat_rect,
                                                    mat_extrinsic
                                                    )
                        tfrecord_writer.write(example.SerializeToString())
                    elif image_set == 'testing':
                        # # !!!! For debugging
                        # # Project velodyne points to image plane
                        # points2D, pointsDist = project_lidar_to_img(calib_dict,
                        #                                             points,
                        #                                             im_height,
                        #                                             im_width)
                        # im_depth_ho = points_to_img(points2D,
                        #                             pointsDist,
                        #                             im_height,
                        #                             im_width)
                        # imsave('data_ex/ho_{}.png'.format(image_set),im_depth_ho)

                        # --- Generate random ratation for decalibration data ---
                        # Generate random vectors for decalibration
                        param_rands = gen_ran_decalib(max_theta,
                                                      max_dist,
                                                      cfg._NUM_GEN)
                        list_im = [im]*cfg._NUM_GEN
                        list_im_depth = [None]*cfg._NUM_GEN

                        for i_ran in xrange(cfg._NUM_GEN):
                            param_decalib = gen_decalib(max_theta,
                                                        max_dist,
                                                        param_rands,
                                                        i_ran)
                            # Copy intrinsic parameters and rotation matrix 
                            # (for reference cam)
                            ran_dict = calib_dict.copy()
                            # Replace extrinsic parameters to decalibrated ones
                            ran_dict[cfg._SET_CALIB[2]] = np.dot(
                                     ran_dict[cfg._SET_CALIB[2]],
                                     quat_to_transmat(param_decalib['q_r'],
                                                      param_decalib['t_vec']))
                    
                            points2D_ran, pointsDist_ran = project_lidar_to_img(
                                                                ran_dict,
                                                                points,
                                                                im_height,
                                                                im_width)
                            list_im_depth[i_ran] = points_to_img(points2D_ran,
                                                                 pointsDist_ran,
                                                                 im_height,
                                                                 im_width)

                            # !!!! For debugging
                            # imsave('data_ex/ho_{}_{}.png'.format(image_set,i_ran),
                            #        list_im_depth[i_ran])
                            # print('  - Angle:{}, nonzero:{}'.format(
                                                # param_decalib['rot'],
                                                # sum(sum(list_im_depth[i_ran]>0)))) 

                        im_placeholder = tf.placeholder(dtype=tf.uint8)
                        im_depth_placeholder = tf.placeholder(dtype=tf.uint8)
                        encoded_image = tf.image.encode_png(im_placeholder)
                        encoded_image_depth = tf.image.encode_png(im_depth_placeholder)

                        if verbose:
                            sys.stdout.write(
                                '... ({}) Writing file to TfRecord {}/{}\n'.format(
                                                    image_set,iter+1,len(imNames)))
                            sys.stdout.flush()

                        png_strings = [sess.run([encoded_image,encoded_image_depth],
                                                 feed_dict={im_placeholder:im,
                                                            im_depth_placeholder:im_depth}) \
                                       for im,im_depth in zip(list_im,list_im_depth)]

                        for i_string in xrange(cfg._NUM_GEN):
                            example = calib_to_tfexample_test(
                                                    png_strings[i_string][0],
                                                    png_strings[i_string][1],
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
                        default = False, type=str2bool)
    parser.add_argument('-is_train', dest='is_train',
                        help='True: Generate training set',
                        default = True, type=str2bool)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print ("Called with args:")
    print (args)
    sys.exit(main(args))