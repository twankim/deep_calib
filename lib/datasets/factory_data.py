# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-06 11:00:57
# @Last Modified by:   twankim
# @Last Modified time: 2017-07-12 13:40:55

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets.dataset_kitti as kitti

dict_data = {
    'kitti': kitti
}

def get_dataset(name, path_data, image_set, reader=None):
    assert name in dict_data,\
                "! Dataset {} is not supported".format(name)
    return dict_data[name].get_data(path_data,image_set)