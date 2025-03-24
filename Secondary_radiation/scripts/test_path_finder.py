#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:18:00 2021

@author: mitchellweikert
"""

import os

cwd = os.getcwd()
print('current working directory is: ' + cwd)
this_path = os.path.realpath(__file__)
print('this path is: ' + this_path)