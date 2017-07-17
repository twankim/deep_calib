# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-06 11:00:57
# @Last Modified by:   twankim
# @Last Modified time: 2017-07-15 17:21:03

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import dataset_kitti as kitti

dict_data = {
    'kitti': kitti
}

def get_dataset(name, path_data, image_set, list_param=[], reader=None):
    """ Returns a dataset

    Args:
        name: String, name of the dataset
        path_data: path to dataset (including dir)
        image_set: String, split name (train/test)
        list_param: list of parameters in the TFRecord file's name
        reader: The subclass of tf.ReaderBase. If left as `None`, 
                then the default reader defined by each dataset is used.

    Returns:
        A 'Dataset' class.
    """
    assert name in dict_data,\
                "! Dataset {} is not supported".format(name)
    return dict_data[name].get_data(
                        path_data,
                        image_set,
                        list_param,
                        reader)