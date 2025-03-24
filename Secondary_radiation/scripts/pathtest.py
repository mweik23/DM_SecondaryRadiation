#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 10:44:30 2021

@author: mitchellweikert
"""

import os

this_path = path_name = os.getcwd()
in_path = this_path.split('Secondary_radiation/')[0]
directories = os.listdir(in_path + 'outputdir/')
dir_bool = os.path.isdir(in_path+'outputdir/'+directories[0])
print(directories)