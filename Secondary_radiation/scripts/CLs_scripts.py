#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:01:52 2021

@author: mitchellweikert
"""

import os
import numpy as np
from scipy import interpolate
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse

#parse command line input
parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='options are save, load, and none', type=str)
args = parser.parse_args()
mode = args.mode

#setup arrays
num_mx = 20
mx_min = 6
mx_max = 500
num_sigmav = 80
log_sigma_min = -27
log_sigma_max = -24
mx_arr = np.round(np.logspace(np.log10(mx_min), np.log10(mx_max), num_mx), 1)
sigmav_arr = np.logspace(log_sigma_min, log_sigma_max, num_sigmav)
num_masks = 10
mask_thicknesses = np.linspace(5.52e-3, 9.65e-3, num_masks)
D0 = 3e28
D0_str = str(D0)
starting_samp = 0
num_samps = 50000
mask_ind=6

#relevant paths
base_path = os.path.realpath(__file__).split('scripts')[0]
dchi_path = base_path + 'dchi_masks/'
dchi_comb_path = base_path + 'dchi_comb_masks/'
dchi_comb_suffix = 'ring_samples_' + str(starting_samp) + '_' + str(starting_samp+num_samps-1)+ \
    '_with_masks_D0_3e28_mx_logspace_'+str(mx_min)+'_'+str(mx_max)+'_'+str(num_mx)+\
        '_round1dec_sigmav_logspace_'+str(log_sigma_min)+'_'+str(log_sigma_max)+'_'+str(num_sigmav)

if type(mask_ind)==int:
    mask_ind_st = mask_ind
    mask_ind_end = mask_ind+1
    num_masks = 1
    mask_thicknesses=np.array([mask_thicknesses[mask_ind]])
else:
    mask_ind_st = 0
    mask_ind_end = num_masks

dchi_comb_suffix += '_mask_thicknesses_'+ '{:.2e}'.format(mask_thicknesses[0])+'_'+'{:.2e}'.format(mask_thicknesses[-1])

if mode=='save':
    #run this cell to load dchi arrays with new file structure 
    #axis 0 : mask thickness, axis 1: mx or D0 pairing, axis 2: sigmav, axis 3: run number

    #create combined delta chi arrays
    dchi_list = os.listdir(dchi_path)

    #initialize file name lists
    dchi_sb_file_names = []
    dchi_b_file_names = []

    #sort dchi file names by mx, D0 and b/sb
    for file in dchi_list:
        if '_ring_' in file:
            if 'dchi_sb' in file:
                dchi_sb_file_names.append(file)
            elif 'dchi_b' in file:
                dchi_b_file_names.append(file)
            else:
                print('formatting of file name not correct: ' + file)

    print('file names sorted')

    test_array = np.load(dchi_path + dchi_sb_file_names[0])
    sh = test_array.shape
    num_mx = sh[1] 
    num_sigmav = sh[2]

    #load arrays
    dchi_sb_tup = tuple(np.load(dchi_path + dchi_sb_file_names[i])[mask_ind_st:mask_ind_end] for i in range(len(dchi_sb_file_names)))
    dchi_b_tup = tuple(np.load(dchi_path + dchi_b_file_names[i])[mask_ind_st:mask_ind_end]  for i in range(len(dchi_b_file_names)))
    print('arrays loaded')
    
    #concatenate arrays
    dchi_sb_comb = np.concatenate(dchi_sb_tup, axis=3)
    del dchi_sb_tup
    dchi_b_comb = np.concatenate(dchi_b_tup, axis=3)
    del dchi_b_tup
    print('arrays have been concatenated')
    
    #save conbined arrays of dchi
    np.save(dchi_comb_path + 'dchi_sb_' + dchi_comb_suffix + '.npy', dchi_sb_comb)
    np.save(dchi_comb_path + 'dchi_b_' + dchi_comb_suffix + '.npy', dchi_b_comb)
    print('combined dchi arrays have been saved')

elif mode=='load':
    #run this cell if dchi arrays have been stitched together
    #load delta chi arrays
    dchi_sb_comb = np.load(dchi_comb_path + 'dchi_sb_' + dchi_comb_suffix + '.npy')
    dchi_b_comb = np.load(dchi_comb_path + 'dchi_b_' + dchi_comb_suffix + '.npy')
    sh = dchi_sb_comb.shape
    num_mx = sh[1]
    num_sigmav = sh[2]
    print('dchi arrays have been loaded successfully')
else:
    print('mode is not recognized')

#sort delta chi arrays and initialize 
dchi_sb_sort = np.sort(dchi_sb_comb)
dchi_b_sort = np.sort(dchi_b_comb)
CLb_heatmap = np.zeros((num_masks, num_mx, num_sigmav))

#find min and max values of delta chi arrays
sb_min = dchi_sb_sort[:, :, :, 0]
sb_max = dchi_sb_sort[:, :, :, -1]
b_min = dchi_b_sort[:, :, :, 0]
b_max = dchi_b_sort[:, :, :, -1]
num_samps = dchi_sb_sort.shape[-1]

#define the CL and find the index of the CL in the limit of serarated distributions
alpha = 0.05
CLb_cut = 0.02/2
p = np.arange(1, num_samps+1)/num_samps
CL = 1-p
ind_alpha = int(np.round((1-alpha)*num_samps))-1

plt_cl=False

#iterate over mx and sigmav finding the delta chi squared corresponding to CL_s = alpha for each mx and sigmav
for m in range(num_masks):
    for i in range(num_mx):
        for j in range(num_sigmav):
            #interpolate CL_sb and CL_b
            CL_sb = interpolate.interp1d(dchi_sb_sort[m, i,-1-j], CL, kind='linear')
            CL_b = interpolate.interp1d(dchi_b_sort[m, i,-1-j], CL, kind='linear')
            #test CL_b
            #dchi_arr = np.linspace(30, 40, 1000)
            #plt.plot(dchi_arr, CL_b(dchi_arr))
            #fig = plt.figure()
            #plt.plot(dchi_b_sort[i,j,-300:], CL[-300:])
            #define a function that returns CL_s-alpha
            def CLs_func(x, *args):
                if np.logical_and(x < args[1][0], x < args[2][0]):
                    CL_s = 1
                elif np.logical_and(x < args[2][0], x<=args[1][1]):
                    CL_s = CL_sb(x)
                elif x<args[2][0]:
                    CL_s = 0
                elif np.logical_and(x <= args[2][1], x < args[1][0]):
                    CL_s = 1/CL_b(x)
                elif np.logical_and(x <= args[2][1], x <= args[1][1]):
                    CL_s = CL_sb(x)/CL_b(x)
                elif x <= args[2][1]:
                    CL_s = 0
                else:
                    CL_s = None

                if CL_s is not None:
                    return CL_s-args[0]
                else:
                    return None
            #Solve for delta chi squared that gives CL_s = alpha. 

            #First appoximate the result iteratively starting in the limit of complete separation of distributions.
            #print('approximating result for ' + str(mx_arr[i]) + ' sigmav = ' + '{:.2e}'.format(sigmav_arr[-1-j]))
            guess0 = dchi_sb_sort[m, i, -1-j, ind_alpha]
            #print('guess0 = ', guess0)
            if guess0 < b_min[m, i,-1-j]:
                guess_final = guess0
            elif guess0 > b_max[m, i,-1-j]:
                guess_final = sb_max[m, i,-1-j]
            else:
                alpha1 = alpha*CL_b(guess0)
                #print('alpha1 = ', alpha1)
                ind_alpha1 = int(np.round((1-alpha1)*num_samps))-1
                guess1 = dchi_sb_sort[m, i, -1-j, ind_alpha1]
                #print('guess1 = ', guess1)
                if guess1 < b_min[m, i,-1-j]:
                    guess_final = guess1
                elif guess1 > b_max[m, i,-1-j]:
                     guess_final = sb_max[m, i,-1-j]
                else: 

                    alpha2 = alpha*CL_b(guess1)
                    #print('alpha2 = ', alpha2)
                    ind_alpha2 = int(np.round((1-alpha2)*num_samps))-1
                    guess2 = dchi_sb_sort[m, i, -1-j, ind_alpha2]
                    #print('guess2 = ', guess2)
                    guess_final = guess2
            #print('final guess is: ', guess_final)       

            #plot to trouble shoot
            if plt_cl:
                dchi_arr = np.linspace(sb_min[m, i, -1-j], b_max[m, i, -1-j], 1000)
                dchi_arr_sb = np.linspace(sb_min[m, i, -1-j], sb_max[m, i, -1-j], 1000)
                dchi_arr_b = np.linspace(b_min[m, i, -1-j], b_max[m, i, -1-j], 1000)
                these_args = (0, (sb_min[m, i, -1-j], sb_max[m, i, -1-j]), (b_min[m, i, -1-j], b_max[m, i, -1-j]))
                CLs_arr = np.array([CLs_func(dchi_arr[k], *these_args) for k in range(len(dchi_arr))])
                fig = plt.figure()
                plt.plot(dchi_arr, CLs_arr, label='CLs')
                plt.plot(dchi_arr_sb, CL_sb(dchi_arr_sb), label='CL_sb')
                plt.plot(dchi_arr_b, CL_b(dchi_arr_b), label='CL_b')
                plt.legend()

            #Then use fsolve to find a more exact solution
            sol = fsolve(CLs_func, guess_final, args=(alpha, (sb_min[m, i,-1-j], sb_max[m, i,-1-j]), (b_min[m, i,-1-j], b_max[m, i,-1-j])))[0]
            #print('solution is given by: ', sol)
            if sol<b_min[m, i,-1-j]:
                res = 1
            elif sol > b_max[m, i,-1-j]:
                res = 0
            else:
                res = CL_b(sol)
            CLb_heatmap[m, i,-1-j] = res
            if res < CLb_cut:
                #print('smallest sigmav value for mx = '+ str(mx_arr[i])+ ' GeV is given by ' + '{:.2e}'.format(sigmav_arr[-1-j]) + ' cm^3/s')
                break

np.save(base_path+'stat_results/' + 'CLb_heatmap_ring_maskthicknesses_'+'{:.2e}'.format(mask_thicknesses[0])+'_'+'{:.2e}'.format(mask_thicknesses[-1])+'.npy', CLb_heatmap)

z_scores = np.linspace(-2, 2, 5)
percentiles = norm.cdf(z_scores)
approx_lims = np.array([[[sigmav_arr[np.where(np.abs(CLb_heatmap[m, i]-percentiles[k])==np.min(np.abs(CLb_heatmap[m, i]-percentiles[k])))[0][0]] \
                for k in range(len(percentiles))] for i in range(num_mx)] for m in range(num_masks)])

sigmav_lims = np.zeros((num_masks, num_mx, len(z_scores)))

for m in range(num_masks):                        
    for i in range(num_mx):
        CLb_int = interpolate.interp1d(sigmav_arr, CLb_heatmap[m, i], kind='cubic')
        def CLb_min_perc(x, *args):
            return CLb_int(x)-args[0]
        for k in range(len(percentiles)):
            print('For mx = ' + str(mx_arr[i]) +  ' and percentile = ' + str(np.round(percentiles[k], 2)) + \
                  ', the approximate value for sigmav_lim is:  ' + '{:.2e}'.format(approx_lims[m, i, k]))
            if approx_lims[m,i,k] == sigmav_arr[-1]:
                sigmav_lims[m,i,k] = sigmav_arr[-1]
                print('sigmav_lim is likely out of range for mask number = ' + str(m) + ', mx = ' + str(mx_arr[i]) + ', percentile = ' + str(percentiles[k]))
            else:
                sigmav_zoom = np.logspace(np.log10(sigmav_arr[0]), min(np.log10(sigmav_arr[-1]), np.log10(3*approx_lims[m, i, k])), 200)
                approx_lim_better = sigmav_zoom[np.where(np.abs(CLb_min_perc(sigmav_zoom, *(percentiles[k],)))==np.min(np.abs(CLb_min_perc(sigmav_zoom, *(percentiles[k],)))))[0][0]]
                print('For mx = ' + str(mx_arr[i]) +  ' and percentile = ' + str(np.round(percentiles[k], 2)) + \
                      ', the better approximate value for sigmav_lim is:  ' + '{:.2e}'.format(approx_lim_better))
                sigmav_lims[m, i, k] = fsolve(CLb_min_perc, approx_lim_better, args=(percentiles[k]))[0]
    fig = plt.figure()
    for k in range(len(percentiles)):
        plt.plot(mx_arr, sigmav_lims[m, :, k], label='CL_b = ' + str(np.round(percentiles[k], 2)))
        plt.xlabel('mx (GeV)')
        plt.ylabel('sigma v (cm^3/s)')
        plt.title('CL_s = 0.05 limits; ring; mask thickness = ' + '{:.2e}'.format(mask_thicknesses[m]))
        plt.legend()
        plt.xscale('log')
        plt.yscale('log')
    plt.savefig(base_path+'stat_results/brazil_CLs_0.05_maskthickness_' + '{:.2e}'.format(mask_thicknesses[m]) + '.pdf')
    plt.close()
