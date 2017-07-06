# -*- coding: utf-8 -*-
# @Author: twankim
# @Date:   2017-07-05 14:26:57
# @Last Modified by:   twankim
# @Last Modified time: 2017-07-06 10:43:33

import os
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_path = os.path.dirname(__file__)

lib_path = os.path.join(this_path,'lib')
add_path(lib_path)