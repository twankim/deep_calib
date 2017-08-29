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
"""Contains model definitions for versions of the Oxford VGG network.

These model definitions were introduced in the following technical report:

  Very Deep Convolutional Networks For Large-Scale Image Recognition
  Karen Simonyan and Andrew Zisserman
  arXiv technical report, 2015
  PDF: http://arxiv.org/pdf/1409.1556.pdf
  ILSVRC 2014 Slides: http://www.robots.ox.ac.uk/~karen/pdf/ILSVRC_2014.pdf
  CC-BY-4.0

More information can be obtained from the VGG website:
www.robots.ox.ac.uk/~vgg/research/very_deep/

Usage:
  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_a(images,lidars)

  with slim.arg_scope(vgg.vgg_arg_scope()):
    outputs, end_points = vgg.vgg_16(images,lidars)

@@vgg_a
@@vgg_16
@@vgg_19
"""
# Modified for deep calibration application
# VGG 16 based deep calibration network

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def last_layer(net,num_preds):
  """ Defines last layer for prediction

  Args:
    net: output of the previous layer. [batch_size, height, width, channels]
    num_preds: number of predictin outputs (integer or dictionary)
            if dictionary, should include following values
            
                num_preds['num_preds']: list of prediction numbers
                num_preds['is_normalize']: corresponding boolean list
  return:
    output tensor with size [batch, 1, 1, number of predictions]

  """
  if isinstance(num_preds,dict):
    # preds = []
    # for i_pred, num_pred in enumerate(num_preds['num_preds']):
    #   if num_preds['is_normalize'][i_pred]:
    #     pred = slim.conv2d(net, num_pred, [1, 1],
    #                        activation_fn=None,
    #                        normalizer_fn=None,
    #                        scope='fc8_{}'.format(i_pred))
    #     preds.append(tf.nn.l2_normalize(pred,dim=3))
    #   else:
    #     preds.append(slim.conv2d(net, num_pred, [1, 1],
    #                              activation_fn=None,
    #                              normalizer_fn=None,
    #                              scope='fc8_{}'.format(i_pred)))
    # net = tf.concat(values=preds,axis=3,name='fc8')
    net = slim.conv2d(net, sum(num_preds['num_preds']), [1, 1],
                      activation_fn=None,
                      normalizer_fn=None,
                      scope='fc8_prev')
    pred_splits = tf.split(net,num_preds['num_preds'],axis=3)
    pred_isnorm = tf.concat([pred_splits[i] for i in xrange(len(num_preds['is_normalize'])) \
                             if num_preds['is_normalize'][i]],axis=3)
    norm_rot = tf.norm(tf.concat(pred_isnorm,axis=3),
                       axis=3,keep_dims=True)
    net = tf.div(net,norm_rot,name='fc8')
  else:
    # net = slim.conv2d(net, num_preds, [1, 1],
    net = slim.conv2d(net, sum(num_preds['num_preds']), [1, 1],
                      activation_fn=None,
                      normalizer_fn=None,
                      scope='fc8')
  return net


def vgg_arg_scope(weight_decay=0.0005):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def vgg_a(images,
          lidars,
          num_preds=7,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='vgg_a',
          fc_conv_padding='VALID'):
  """Oxford Net VGG 11-Layers version A Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    images: a tensor of size [batch_size, height, width, channels].
    lidars: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_a', [images,lidars]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      # ConvNets for image
      net = slim.repeat(images, 1, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 1, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 2, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      # net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
      # net = slim.max_pool2d(net, [2, 2], scope='pool4')
      # net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
      # net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # ConvNets for lidar
      net2 = slim.repeat(lidars, 1, slim.conv2d, 32, [3, 3], scope='conv1_lidar')
      net2 = slim.max_pool2d(net2, [2, 2], scope='pool1_lidar')
      net2 = slim.repeat(net2, 1, slim.conv2d, 64, [3, 3], scope='conv2_lidar')
      net2 = slim.max_pool2d(net2, [2, 2], scope='pool2_lidar')
      net2 = slim.repeat(net2, 2, slim.conv2d, 128, [3, 3], scope='conv3_lidar')
      net2 = slim.max_pool2d(net2, [2, 2], scope='pool3_lidar')
      # net2 = slim.repeat(net2, 2, slim.conv2d, 512, [3, 3], scope='conv4_lidar')
      # net2 = slim.max_pool2d(net2, [2, 2], scope='pool4_lidar')
      # net2 = slim.repeat(net2, 2, slim.conv2d, 512, [3, 3], scope='conv5_lidar')
      # net2 = slim.max_pool2d(net2, [2, 2], scope='pool5_lidar')

      # Concat two channels
      net = tf.concat(values=[net,net2],axis=3)

      with tf.variable_scope('match_feat'):
        # Remaining ConvNets for Feature Matching
        net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 2, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

      with tf.variable_scope('regression'):
        # Use conv2d instead of fully_connected layers.
        net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = last_layer(net,num_preds)

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
          end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_a.default_image_size = 224


def vgg_16(images,
           lidars,
           num_preds=7,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           fc_conv_padding='VALID'):
  """Oxford Net VGG 16-Layers version D Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    images: a tensor of size [batch_size, height, width, channels].
    lidars: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_16', [images,lidars]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      # net = slim.max_pool2d(net, [2, 2], scope='pool4')
      # net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      # net = slim.max_pool2d(net, [2, 2], scope='pool5')

      with tf.variable_scope('lidar_feat'):
        # ConvNets for lidar
        # net2 = slim.max_pool2d(lidars, [2, 2], scope='pool0_lidar')
        # net2 = slim.repeat(net2, 2, slim.conv2d, 32, [3, 3], scope='conv1_lidar')

        net2 = slim.repeat(lidars, 2, slim.conv2d, 32, [3, 3], scope='conv1_lidar')

        net2 = slim.max_pool2d(net2, [2, 2], scope='pool1_lidar')
        net2 = slim.repeat(net2, 2, slim.conv2d, 64, [3, 3], scope='conv2_lidar')
        net2 = slim.max_pool2d(net2, [2, 2], scope='pool2_lidar')
        net2 = slim.repeat(net2, 3, slim.conv2d, 128, [3, 3], scope='conv3_lidar')
        net2 = slim.max_pool2d(net2, [2, 2], scope='pool3_lidar')
        # net2 = slim.repeat(net2, 2, slim.conv2d, 512, [3, 3], scope='conv4_lidar')
        # net2 = slim.max_pool2d(net2, [2, 2], scope='pool4_lidar')
        # net2 = slim.repeat(net2, 2, slim.conv2d, 512, [3, 3], scope='conv5_lidar')
        # net2 = slim.max_pool2d(net2, [2, 2], scope='pool5_lidar')

      with tf.variable_scope('match_feat'):
        # Concat two channels
        net = tf.concat(values=[net,net2],axis=3)
        # Remaining ConvNets for Feature Matching
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

      with tf.variable_scope('regression'):
        # Use conv2d instead of fully_connected layers.
        # net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
        net = slim.conv2d(net, 512, [7, 7], padding=fc_conv_padding, scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        # net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.conv2d(net, 256, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        # net = last_layer(net,num_preds)
        net = slim.conv2d(net, sum(num_preds['num_preds']), [1, 1],
                      activation_fn=None,
                      normalizer_fn=None,
                      scope='fc8')

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
          end_points[sc.name + '/fc8'] = net
      
      return net, end_points
vgg_16.default_image_size = 224


def vgg_19(images,
           lidars,
           num_preds=7,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_19',
           fc_conv_padding='VALID'):
  """Oxford Net VGG 19-Layers version E Example.

  Note: All the fully_connected layers have been transformed to conv2d layers.
        To use in classification mode, resize input to 224x224.

  Args:
    images: a tensor of size [batch_size, height, width, channels].
    lidars: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    dropout_keep_prob: the probability that activations are kept in the dropout
      layers during training.
    spatial_squeeze: whether or not should squeeze the spatial dimensions of the
      outputs. Useful to remove unnecessary dimensions for classification.
    scope: Optional scope for the variables.
    fc_conv_padding: the type of padding to use for the fully connected layer
      that is implemented as a convolutional layer. Use 'SAME' padding if you
      are applying the network in a fully convolutional manner and want to
      get a prediction map downsampled by a factor of 32 as an output. Otherwise,
      the output prediction map will be (input / 32) - 6 in case of 'VALID' padding.

  Returns:
    the last op containing the log predictions and end_points dict.
  """
  with tf.variable_scope(scope, 'vgg_19', [images,lidars]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      # net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
      # net = slim.max_pool2d(net, [2, 2], scope='pool4')
      # net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
      # net = slim.max_pool2d(net, [2, 2], scope='pool5')

      # ConvNets for lidar
      net2 = slim.repeat(lidars, 2, slim.conv2d, 32, [3, 3], scope='conv1_lidar')
      net2 = slim.max_pool2d(net2, [2, 2], scope='pool1_lidar')
      net2 = slim.repeat(net2, 2, slim.conv2d, 64, [3, 3], scope='conv2_lidar')
      net2 = slim.max_pool2d(net2, [2, 2], scope='pool2_lidar')
      net2 = slim.repeat(net2, 4, slim.conv2d, 128, [3, 3], scope='conv3_lidar')
      net2 = slim.max_pool2d(net2, [2, 2], scope='pool3_lidar')
      # net2 = slim.repeat(net2, 2, slim.conv2d, 512, [3, 3], scope='conv4_lidar')
      # net2 = slim.max_pool2d(net2, [2, 2], scope='pool4_lidar')
      # net2 = slim.repeat(net2, 2, slim.conv2d, 512, [3, 3], scope='conv5_lidar')
      # net2 = slim.max_pool2d(net2, [2, 2], scope='pool5_lidar')

      # Concat two channels
      net = tf.concat(values=[net,net2],axis=3)

      with tf.variable_scope('match_feat'):
        # Remaining ConvNets for Feature Matching
        net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
        net = slim.max_pool2d(net, [2, 2], scope='pool4')
        net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

      with tf.variable_scope('regression'):
        # Use conv2d instead of fully_connected layers.
        net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout6')
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = last_layer(net,num_preds)

        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
          end_points[sc.name + '/fc8'] = net
      return net, end_points
vgg_19.default_image_size = 224

# Alias
vgg_d = vgg_16
vgg_e = vgg_19
