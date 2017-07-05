# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-05 14:26:57
# @Last Modified by:   twankim
# @Last Modified time: 2017-07-05 14:27:08

import os

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


curr_path = os.getcwd()

lib_path = os.path.join(curr_path,'lib')
add_path(lib_path)