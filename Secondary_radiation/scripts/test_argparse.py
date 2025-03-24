#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:44:11 2021

@author: mitchellweikert
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mx', help = 'Dark Matter Mass (GeV)', type = float)
parser.add_argument('--sigma_v', help = 'Dark Matter Cross Section (cm^3/s)', type=float)
args = parser.parse_args()
print('Dark matter mass = ', str(args.mx))
print('Dark matter cross section = ', str(args.sigma_v))