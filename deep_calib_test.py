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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

import _init_paths
from datasets import factory_data
from nets import factory_nets
from preprocessing import preprocessing_factory
from datasets.utils_dataset import tf_prepare_test

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/data/tf/kitti_calib/checkpoints/vgg_16/weight_1',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', None, 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'kitti', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

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

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = factory_data.get_dataset(
            FLAGS.dataset_name,
            FLAGS.dataset_dir,
            'test',
            FLAGS.list_param.split(','))

    ####################
    # Select the model #
    ####################
    network_fn = factory_nets.get_network_fn(
        FLAGS.model_name,
        num_preds=dataset.num_preds,
        is_training=False)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=2 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)
    [image,lidar,y_true,params_crop] = provider.get(['image','lidar','y',
                                                     'params_crop'])

    # Crop image and lidar to consider only sensed region
    image,lidar = tf_prepare_test(image,lidar,params_crop)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
    preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=False)

    lidar_pool = [int(l_i) for l_i in FLAGS.lidar_pool.split(',')]

    test_image_size = FLAGS.eval_image_size or network_fn.default_image_size

    image,lidar = preprocessing_fn(image,lidar,
                                   test_image_size,
                                   test_image_size,
                                   pool_size=lidar_pool)

    images, lidars, y_trues = tf.train.batch(
        [image, lidar, y_true],
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=5 * FLAGS.batch_size)

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

    if FLAGS.weight_loss:
      weight_loss = FLAGS.weight_loss
      weights_preds = np.ones(sum(dataset.num_preds['num_preds']))
      i_reg_start = 0
      for i_reg,is_normalize in enumerate(dataset.num_preds['is_normalize']):
        num_preds = dataset.num_preds['num_preds'][i_reg]
        if is_normalize:
          weights_preds[i_reg_start:i_reg_start+num_preds] = FLAGS.weight_loss
        i_reg_start += num_preds
      weights_preds = tf.constant(np.tile(weights_preds,(FLAGS.batch_size,1)))
    else:
      weight_loss = 1
      weights_preds = 1.0

    # Define the metrics:
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
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    if FLAGS.eval_dir:
      path_log = FLAGS.eval_dir
    else:
      path_log = checkpoint_path

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=path_log,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
