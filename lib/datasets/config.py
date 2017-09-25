# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-05 13:38:36
# @Last Modified by:   twankim
# @Last Modified time: 2017-09-25 11:21:25

# Configureation file for dataset
from easydict import EasyDict as edict

__C = edict()

cfg = __C

__C._TASK = 'object'

__C._TYPE_CALIB = 'calib'
__C._TYPE_IMAGE = 'image_2'
__C._TYPE_VELO = 'velodyne'

__C._FORMAT_CALIB = '.txt'
__C._FORMAT_IMAGE = '.png'
__C._FORMAT_VELO = '.bin'

__C._DIR_CALIB = 'data_{}_{}'.format(__C._TASK,__C._TYPE_CALIB)
__C._DIR_IMAGE = 'data_{}_{}'.format(__C._TASK,__C._TYPE_IMAGE)
__C._DIR_VELO = 'data_{}_{}'.format(__C._TASK,__C._TYPE_VELO)

__C._NUM_GEN = 5 # Number of random generation per image

__C._SET_CALIB = ['P2','R0_rect','Tr_velo_to_cam']

__C._TF_FORMAT = '{}_calib_{:02d}_{:.1f}_{}.tfrecord'