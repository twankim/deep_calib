from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from skimage.io import (imread,imsave)
import glob
import math
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

import _init_paths
from datasets.config import cfg
from datasets.dataset_kitti import (get_calib_mat,_NUM_PREDS)
from datasets.utils_dataset import *
from nets import factory_nets
from preprocessing import preprocessing_factory

class Predictor:
    def __init__(self,model_name,preprocessing_name,checkpoint_path,
                 test_image_size,lidar_pool,is_crop):
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.preprocessing_name = preprocessing_name
        self.image_preprocessing_fn = preprocessing_factory.get_preprocessing(
                                            self.preprocessing_name,
                                            is_training=False)
        self.network_fn = factory_nets.get_network_fn(
                            self.model_name,
                            num_preds=_NUM_PREDS,
                            is_training=False)
        self.test_image_size = test_image_size or self.network_fn.default_image_size
        self.graph = tf.Graph()
        self.load_model()
        self.lidar_pool = lidar_pool
        self.params_crop = None
        self.is_crop = is_crop

    def load_model(self):
        self.im_placeholder = tf.placeholder(dtype=tf.uint8,
                                                shape=[None,None,3])
        self.im_depth_placeholder = tf.placeholder(dtype=tf.uint8,
                                                      shape=[None,None,1])

        with slim.arg_scope(factory_nets.arg_scopes_map[self.model_name]()):
            # Crop image and lidar to consider only sensed region
            image,lidar = tf_prepare_test(self.im_placeholder,
                                          self.im_depth_placeholder,
                                          self.params_crop)
            image,lidar = self.preprocessing_fn(image,lidar,
                                                self.test_image_size,
                                                self.test_image_size,
                                                pool_size=self.lidar_pool)

            # Change format to [batch_size, height, width, channels]
            images = tf.expand_dims(image, 0)
            lidars = tf.expand_dims(lidar, 0)

            self.y_preds, _ = network_fn(images,lidars)
        
        self.sess = tf.Session()
        # self.saver = tf.train.import_meta_graph(self.checkpoint_path+'.meta')
        saver = tf.train.Saver()
        saver.restore(self.sess,self.checkpoint_path)

    def predict(self,im,im_lidar):

        y_preds_val = self.sess.run(self.y_preds,
                                    feed_dict={self.im_placeholder:im,
                                               self.im_depth_placeholder:im_lidar
                                               })
        y_preds_val = np.squeeze(y_preds_val,axis=0)
        # Normalize quaternion to have unit norm
        q_r_preds = yr_to_qr(y_preds_val[:4],max_theta)

        print('Norm_previous:{}'.format(np.linalg.norm(q_r_preds)))
        q_r_preds = q_r_preds/np.linalg.norm(q_r_preds)

        if self.is_crop:
            img_temp,lidar_temp = self.sess.run(
                            image,lidar,
                            feed_dict={self.im_placeholder:im,
                                       self.im_depth_placeholder:im_lidar
                            })

            _R_MEAN = 123.68
            _G_MEAN = 116.78
            _B_MEAN = 103.94
            _BW_MEAN = (_R_MEAN+_G_MEAN+_B_MEAN)/3.0
            img_temp[:,:,0] += _R_MEAN
            img_temp[:,:,1] += _G_MEAN
            img_temp[:,:,2] += _B_MEAN
            lidar_temp += _BW_MEAN

            img_temp = img_temp.astype(np.uint8)
            lidar_temp = np.squeeze(lidar_temp.astype(np.uint8),axis=2)

            return y_preds_val,q_r_preds,img_temp,lidar_temp
        else:
            return y_preds_val,q_r_preds

    @staticmethod
    def calibrate(ran_dict,q_r_preds,y_preds_val,points,im_height,im_width):
        # Calibarte based on the prediction
        cal_dict = ran_dict.copy()

        Rt = quat_to_transmat(q_r_preds,y_preds_val[4:])
        Rt_cal = Rt.copy()
        Rt_cal[:3,:3] = Rt[:3,:3].T
        Rt_cal[:3,3] = -np.dot(Rt[:3,:3].T,Rt[:3,3])

        cal_dict[cfg._SET_CALIB[2]] = np.dot(
              ran_dict[cfg._SET_CALIB[2]],Rt_cal)

        points2D_cal, pointsDist_cal = project_lidar_to_img(cal_dict,
                                                            points,
                                                            im_height,
                                                            im_width)

        return points2D_cal, pointsDist_cal