#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 16:39:27 2021

@author: mitchellweikert
"""

import numpy as np
import time
delta = 0.025
N1 = 60
N2 = 30
half1 = int((N1-1)/2)
half2 = int((N2-1)/2)

start = time.time()
th1 = delta*np.linspace(-half1, half1, N1)
th2 = delta*np.linspace(-half2, half2, N2)
th1_ext = np.zeros(N1*N2)
th2_ext = np.zeros(N1*N2)
for i in range(N2):
    th1_ext[i*N1:(i+1)*N1] = th1
    th2_ext[i*N1:(i+1)*N1] = th2[i]*np.ones(N1)
end = time.time()
print('The time to create the extended coordinate vectors was: ' + str(end-start) + ' s')
start = time.time()
TH1_row, TH1_col = np.meshgrid(th1_ext, th1_ext)
TH2_row, TH2_col = np.meshgrid(th2_ext, th2_ext)
end = time.time()
print('The time to create the extended coordinate matrices was: ' + str(end-start) + ' s')
rms = .25
cor_len = 0.0125

start = time.time()
dless_Sigma = np.exp(-(1/(4*cor_len**2))*((TH1_row-TH1_col)**2+(TH2_row-TH2_col)**2))
end = time.time()
print('The time to create the correlation matrix was: ' + str(end-start) + ' s')
gen = np.identity(N1*N2)-dless_Sigma
start = time.time()
gen_sq = np.matmul(gen, gen)
end = time.time()
print('The time to square the correlation matrix was: ' + str(end-start) + ' s')
gen3 = np.matmul(gen, gen_sq)

gen4 = np.matmul(gen_sq, gen_sq)