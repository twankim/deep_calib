# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications copyright (C) 2017 UT Austin/Taewan Kim
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""
# Modified for deep learning based calibration code

# Code for real application on KITTI
# Input: Image, LIDAR, calib file (P2, Rect0), initial guess (H_init)
# Output: Result of calibration (image, H)

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

tf.app.flags.DEFINE_integer(
    'batch_size', 1, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_integer(
    'is_rand', True, 'Turn on random decalibration')

tf.app.flags.DEFINE_integer(
    'num_gen', 5, 'Number of random decalibs to generate')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/data/tf/kitti_calib/checkpoints/vgg_16/weight_1',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'kitti', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dir_image', None, 'The directory where the image files are stored.')

tf.app.flags.DEFINE_string(
    'dir_lidar', None, 'The directory where the lidar files are stored.')

tf.app.flags.DEFINE_string(
    'dir_calib', None, 'The directory where the calibration files are stored.')

tf.app.flags.DEFINE_string(
    'dir_out', None, 'The directory where the output files are stored.')

tf.app.flags.DEFINE_string(
    'format_image', 'png', 'The format of image. default=png')

tf.app.flags.DEFINE_string(
    'format_lidar', 'bin', 'The format of lidar. default=bin')

tf.app.flags.DEFINE_string(
    'format_calib', 'txt', 'The format of calibartion file. default=txt')

tf.app.flags.DEFINE_string(
    'list_param', '20,1.5',
    'List of parameters for the train/test data. max_rotation,max_translation')

tf.app.flags.DEFINE_string(
    'lidar_pool', None,
    'Kernel size for Max-pooling LIDAR Image: height,width. defualt=5,2')

tf.app.flags.DEFINE_string(
    'model_name', 'vgg_16', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_string(
    'weight_loss', None,
    'The weight to balance predictions. ex) multiplied to the rotation quaternion')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_boolean(
    'is_crop', True, 'Eval image size')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dir_image:
    raise ValueError('You must supply the image directory with --dir_image')
  if not FLAGS.dir_lidar:
    raise ValueError('You must supply the lidar directory with --dir_lidar')
  if not FLAGS.dir_calib:
    raise ValueError('You must supply the calibration directory with --dir_calib')

  # Parameters for random generation
  max_theta,max_dist = map(float,FLAGS.list_param.split(','))

  tf.logging.set_verbosity(tf.logging.INFO)

  # Get the list of images to process
  imList = glob.glob(os.path.join(FLAGS.dir_image,'*.'+FLAGS.format_image))
  imList.sort()
  imNames = [os.path.split(pp)[1].strip('.{}'.format(FLAGS.format_image)) \
             for pp in imList]

  for iter,imName in enumerate(imNames):
    decalibs_gt = []
    decalibs_pred = []
    decalibs_qr_gt = []
    decalibs_qr_pred = []

    # Get original calibration info
    f_calib = os.path.join(FLAGS.dir_calib,imName+'.'+FLAGS.format_calib)
    temp_dict = get_calib_mat(f_calib)

    # Read lidar points
    f_lidar = os.path.join(FLAGS.dir_lidar,imName+'.'+FLAGS.format_lidar)
    points_org = np.fromfile(f_lidar,dtype=np.float32).reshape(-1,4)
    points = points_org[:,:3] # exclude reflectance

    # Read image file
    f_image = os.path.join(FLAGS.dir_image,imName+'.'+FLAGS.format_image)
    im = imread(f_image)
    im_height,im_width = np.shape(im)[0:2]

    # Project velodyne points to image plane
    points2D, pointsDist = project_lidar_to_img(temp_dict,
                                                points,
                                                im_height,
                                                im_width)

    # Write as one image (Ground truth)
    im_depth,_ = points_to_img(points2D,pointsDist,im_height,im_width)
    f_res_im = os.path.join(FLAGS.dir_out,'{}_gt.{}'.format(
                                  imName,FLAGS.format_image))

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)

    lidar_pool = [int(l_i) for l_i in FLAGS.lidar_pool.split(',')]

    ####################
    # Select the model #
    ####################
    network_fn = factory_nets.get_network_fn(
        FLAGS.model_name,
        num_preds=_NUM_PREDS,
        is_training=False)

    # Randomly generate dealibration
    param_rands = gen_ran_decalib(max_theta,max_dist,FLAGS.num_gen)
    for i_ran in xrange(FLAGS.num_gen):
      with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()

        param_decalib = gen_decalib(max_theta,max_dist,param_rands,i_ran)
        ran_dict = temp_dict.copy()
        ran_dict[cfg._SET_CALIB[2]] = np.dot(
                ran_dict[cfg._SET_CALIB[2]],
                quat_to_transmat(param_decalib['q_r'],param_decalib['t_vec']))

        points2D_ran, pointsDist_ran = project_lidar_to_img(ran_dict,
                                                            points,
                                                            im_height,
                                                            im_width)

        # Write before the calibration
        im_depth_ran, params_crop = points_to_img(points2D_ran,
                                                  pointsDist_ran,
                                                  im_height,
                                                  im_width)
        f_res_im_ran = os.path.join(FLAGS.dir_out,'{}_rand{}.{}'.format(
                                    imName,i_ran,FLAGS.format_image))
        # Save ground truth decalibration
        decalibs_gt.append(param_decalib['y'])
        decalibs_qr_gt.append(param_decalib['q_r'])

        # ---------- Prediction of y (decalibration) ----------
        # To TF format
        # im_placeholder = tf.placeholder(dtype=tf.uint8)
        # im_depth_placeholder = tf.placeholder(dtype=tf.uint8)
        im_placeholder = tf.constant(im)
        im_depth_placeholder = tf.constant(im_depth_ran)
        # encoded_image = tf.image.encode_png(im_placeholder)
        # encoded_image_depth = tf.image.encode_png(im_depth_placeholder)

        # Crop image and lidar to consider only sensed region
        image,lidar = tf_prepare_test(im_placeholder,im_depth_placeholder,
                                      params_crop)

        test_image_size = FLAGS.eval_image_size or network_fn.default_image_size

        image,lidar = preprocessing_fn(image,lidar,
                                       test_image_size,
                                       test_image_size,
                                       pool_size=lidar_pool)

        # For debugging
        # Check actual patches provided to the network
        if FLAGS.is_crop:
          with tf.Session('') as sess:
            lidar_temp = sess.run(lidar)
            img_temp = sess.run(image)
            
            _R_MEAN = 123.68
            _G_MEAN = 116.78
            _B_MEAN = 103.94
            _BW_MEAN = (_R_MEAN+_G_MEAN+_B_MEAN)/3.0
            img_temp[:,:,0] += _R_MEAN
            img_temp[:,:,1] += _G_MEAN
            img_temp[:,:,2] += _B_MEAN
            lidar_temp += _BW_MEAN

            path_crop = os.path.join(FLAGS.dir_out,'crops')
            if not os.path.exists(path_crop):
              os.makedirs(path_crop)
            if i_ran==0:
              imsave(os.path.join(path_crop,'{}_rgb_org.png'.format(imName)),im)

            crop_name = os.path.join(path_crop,'{}_{}'.format(imName,i_ran))

            imsave(crop_name+'_rgb.png',img_temp.astype(np.uint8))
            imsave(crop_name+'_lidar.png',np.squeeze(lidar_temp.astype(np.uint8),
                                                     axis=2))
            imsave(crop_name+'_lidar_org.png',np.squeeze(im_depth_ran,axis=2))

        # Change format to [batch_size, height, width, channels]
        # batch_size = 1
        images = tf.expand_dims(image, 0)
        lidars = tf.expand_dims(lidar, 0)

        ####################
        # Define the model #
        ####################
        y_preds, _ = network_fn(images,lidars)

        if FLAGS.moving_average_decay:
          variable_averages = tf.train.ExponentialMovingAverage(
              FLAGS.moving_average_decay, tf_global_step)
          variables_to_restore = variable_averages.variables_to_restore(
              slim.get_model_variables())
          variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
          variables_to_restore = slim.get_variables_to_restore()

        # predictions = tf.argmax(logits, 1)
        # labels = tf.squeeze(labels)

        if FLAGS.weight_loss:
          weight_loss = FLAGS.weight_loss
          weights_preds = np.ones(sum(_NUM_PREDS['num_preds']))
          i_reg_start = 0
          for i_reg,is_normalize in enumerate(_NUM_PREDS['is_normalize']):
            num_preds = _NUM_PREDS['num_preds'][i_reg]
            if is_normalize:
              weights_preds[i_reg_start:i_reg_start+num_preds] = FLAGS.weight_loss
            i_reg_start += num_preds
          weights_preds = tf.constant(np.tile(weights_preds,(FLAGS.batch_size,1)))
        else:
          weight_loss = 1
          weights_preds = 1.0

        # Define the metrics:
        y_trues = tf.constant(np.expand_dims(param_decalib['y'].copy(),0),
                              dtype=tf.float32)
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
          'MSE': slim.metrics.streaming_mean_squared_error(
                                  y_preds, y_trues),
          'MSE_{}'.format(weight_loss): slim.metrics.streaming_mean_squared_error(
                                  y_preds, y_trues, weights=weights_preds),
        })

        # Print the summaries to screen.
        for name, value in names_to_values.items():
          summary_name = 'eval/%s' % name
          op = tf.summary.scalar(summary_name, value, collections=[])
          op = tf.Print(op, [value], summary_name)
          tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # TODO(sguada) use num_epochs=1
        if FLAGS.max_num_batches:
          num_batches = FLAGS.max_num_batches
        else:
          # This ensures that we make a single pass over all of the data.
          num_batches = math.ceil(1 / float(FLAGS.batch_size))

        if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
          checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
        else:
          checkpoint_path = FLAGS.checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)

        path_log = FLAGS.dir_out

        y_preds_val = slim.evaluation.evaluate_once(
            master=FLAGS.master,
            checkpoint_path=checkpoint_path,
            logdir=path_log,
            num_evals=num_batches,
            eval_op=list(names_to_updates.values()),
            final_op=y_preds,
            variables_to_restore=variables_to_restore)

      y_preds_val = np.squeeze(y_preds_val,axis=0)
      # Normalize quaternion to have unit norm
      q_r_preds = yr_to_qr(y_preds_val[:4],max_theta)

      print('Norm_previous:{}'.format(np.linalg.norm(q_r_preds)))
      q_r_preds = q_r_preds/np.linalg.norm(q_r_preds)

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
      # Save predicted decalibration
      decalibs_pred.append(y_preds_val)
      decalibs_qr_pred.append(q_r_preds)

      # Write after the calibration
      im_depth_cal,_ = points_to_img(points2D_cal,
                                     pointsDist_cal,
                                     im_height,
                                     im_width)
      f_res_im_cal = os.path.join(FLAGS.dir_out,'{}_cal{}.{}'.format(
                                  imName,i_ran,FLAGS.format_image))

      imlidarwrite(f_res_im_cal,im,im_depth_cal)

      imlidarwrite(f_res_im_ran,im,im_depth_ran)
      if i_ran==0:
        imlidarwrite(f_res_im,im,im_depth)
         
    # write 7vec, MSE as txt file
    # decalibs_pred, decalibs_gt

    with open(os.path.join(FLAGS.dir_out,imName+'_res.txt'),'w') as f_res:
      for i_ran,(vec_gt,vec_pred) in enumerate(zip(decalibs_gt,decalibs_pred)):
        f_res.write('i_ran:{},   gt:{}\n'.format(i_ran,vec_gt))
        f_res.write('i_ran:{}, pred:{}\n'.format(i_ran,vec_pred))
        mse_val = ((vec_gt - vec_pred)**2).mean()
        mse_rot = ((vec_gt[:4]-vec_pred[:4])**2).mean()
        mse_tran = ((vec_gt[4:]-vec_pred[4:])**2).mean()
        f_res.write('i_ran:{}, MSE:{}, MSE_rot:{}, MSE_trans:{}\n'.format(
                    i_ran,mse_val,mse_rot,mse_tran))

    with open(os.path.join(FLAGS.dir_out,imName+'_res_qr.txt'),'w') as f_res:
      for i_ran,(qr_gt,qr_pred) in enumerate(zip(decalibs_qr_gt,decalibs_qr_pred)):
        f_res.write('i_ran:{},   gt:{}\n'.format(i_ran,qr_gt))
        f_res.write('i_ran:{}, pred:{}\n'.format(i_ran,qr_pred))
        mse_qr = ((qr_gt-qr_pred)**2).mean()
        f_res.write('i_ran:{}, MSE_qr:{}\n'.format(i_ran,mse_qr))

if __name__ == '__main__':
  tf.app.run()