#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 15:01:32 2021

@author: mitchellweikert
"""
import numpy as np

#number of steps
nr = 800
nE = 400
nt = 100 #guess
nnu = 3
nx = 58
ny = 20

#set up unit grid
def grid(nr, nE, r_range, E_range, E_spacing='lin'):
    rmin = r_range[0]
    rmax = r_range[1]
    Emin = E_range[0]
    Emax = E_range[1]
    r = np.linspace(rmin, rmax, nr)
    dr = np.array([r[i+1]-r[i] for i in range(nr-1)])
    if E_spacing=='lin':
        E = np.linspace(Emin, Emax, nE)
        dE = np.array([E[i+1]-E[i] for i in range(nE-1)])
        E = E[0:-1]
        E + dE[0]/2
    if E_spacing=='log':
        E = np.logspace(np.log10(Emin), np.log10(Emax), nE)
        dE = np.array([E[i+1]-E[i] for i in range(nE-1)])
    rr, EE = np.meshgrid(r, E)
    return rr, EE, dr, dE

   
