# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-06-21 11:55:51
# @Last Modified by:   twankim
# @Last Modified time: 2017-07-11 14:04:15

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
from datasets import factory_data
from datasets.config import cfg
from deployment import model_deploy
from preprocessing import preprocessing_factory as factory_preprocessing

def main(args):
    # Choose GPU ID specified 
    if not args.is_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gid
    else:

    f_train = args.f_train
    d_name = args.name
    
    # Path to directory saving tfrecord files
    path_tf = os.path.join(args.path_tf,'{}_calib'.format(d_name))
    
    # Path of tf record file
    path_data = os.path.join(path_tf,f_train)
    
    # Path to save checpoint files
    path_cp = os.path.join(path_tf,f_train,'checkpoints')
    if not os.path.exists(path_cp):
        os.makedirs(path_cp)

    batch_size = args.batch_size

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        # ----------- Create global step -----------
        global_step = slim.create_global_step()

        # ----------- Load dataset -----------
        dataset = factory_data.get_dataset(d_name,path_data,'train')

        # ----------- Select network -----------

        # ----------- Select the preprocessing function -----------
        image_preprocessing_fn = factory_preprocessing.get_preprocessing(
                                    args.model_name,
                                    is_training=True)
        
        # ----------- Create dataset provider -----------
        provider = slim.dataset_data_provider.DatasetDataProvider(
                        dataset,
                        common_queue_capacity=20*batch_size,
                        common_queue_min=10*batch_size)
        [image,lidar,y_true] = provider.get(['image','lidar','y'])
        train_image_size = args.imsize or fn_network.default_image_size
        image = image_preprocessing_fn(image,train_image_size,train_image_size)
        lidar = image_preprocessing_fn(lidar,train_image_size,train_image_size)
        [images,lidars,y_trues] = tf.train.batch(
                        [image,lidar,y_true],
                        batch_size = batch_size,
                        capacity=5*batch_size)
        batch_queue = slim.prefetch_queue.prefetch_queue(
                        [images,lidars,y_trues], capacity=2)

    # ----------- Define the model -----------
    def fn_model(batch_queue):
        images, lidars, y_trues = batch_queue.dequeue()
        y_preds, end_points = fn_network(images,lidars)

        # ----------- Specify the loss function -----------
        tf.losses.mean_squared_error(
                        labels=y_trues,
                        predictions=y_preds,
                        weights=1.0)
        return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
    # Gather update_ops
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Add summaries for end_points
    end_points


def parse_args():
    def str2bool(v):
        return v.lower() in ('true', '1')

    parser = argparse.ArgumentParser(description=
                        'Calibration with deep learning')
    # Currently support only one GPU ID
    parser.add_argument('-gid', dest='gid',
                        help='CUDA_VISIBLE_DEVICES (Check your machine ID and ex) 1',
                        default = '0', type = str)
    parser.add_argument('-is_cpu', dest='is_cpu',
                        help='Use CPU only. True/False',
                        default = False, type = str2bool)
    parser.add_argument('-name', dest='name',
                        hel='Name of Dataset',
                        default = 'kitti', type = str)
    parser.add_argument('-data', dest='path_tf',
                        help='Path to dataset TfRecords',
                        default = '/data/tf',
                        type = str)
    parser.add_argument('-f_train', dest='f_train',
                        help='Name of TfRecord/Checkpoint (excluding .tfrecord)',
                        default = 'kitti_calib_20_1.5_train', type = str)
    parser.add_argument('-batch', dest='batch_size',
                        help='Number of samples in each batch',
                        default = 32, type = int)
    parser.add_argument('-imsize', dest='imsize',
                        help='imsize x imsize: image size for Training',
                        default = None, type = int)
    parser.add_argument('-model', dest='model_name',
                        help='Name of convnet model to use',
                        default = 'vgg', type = str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print "Called with args:"
    print args
    sys.exit(main(args))