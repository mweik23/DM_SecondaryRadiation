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
parser.add_argument('--ellipse_params', help='these are the parameters of the ellipse of best fit ordered as [w_r, r, a, sigma_a]', nargs='+', type=float)
args = parser.parse_args()
mode = args.mode

#setup arrays                                                                                                                       
num_mx = 20
mx_min = 6
mx_max = 500
num_sigmav = 80
num_masks = 10
log_sigma_min = -27
log_sigma_max = -24
mx_arr = np.round(np.logspace(np.log10(mx_min), np.log10(mx_max), num_mx), 1)
sigmav_arr = np.logspace(log_sigma_min, log_sigma_max, num_sigmav)
mask_thicknesses = np.linspace(0.5*args.ellipse_params[3], 3*args.ellipse_params[3], num_masks)
D0 = 3e28
D0_str = str(D0)

type1 = 'b_ring'
type2 = 'b_noring'

starting_samp1 = 50000
starting_samp2 = 0
num_samps_tot = 50000
num_samps_batch = 500

#relevant paths                                                                                                                    
base_path = os.path.realpath(__file__).split('scripts')[0]
dchi_path_general = base_path + 'dchi_masks'
dchi_comb_path = base_path + 'dchi_comb_masks/'
dchi_comb_suffix = '_with_masks_'+'_D0_'+str(D0)+'_mx_logspace_'+str(mx_min)+'_'+str(mx_max)+'_'+str(num_mx)+\
        '_sigmav_logspace_'+str(log_sigma_min)+'_'+str(log_sigma_max)+'_'+str(num_sigmav)+ '_mask_thicknesses_+'+ '{:.2e}'.format(mask_thicknesses[0]) +'_'+'{:.2e}'.format(mask_thicknesses[-1]) +'.npy'
if 'noring' in type1:
    dchi_path1 = dchi_path_general + '_noring/'
else:
    dchi_path1 = dchi_path_general + '/'

if 'noring' in type2:
    dchi_path2 = dchi_path_general + '_noring/'
else:
    dchi_path2 = dchi_path_general + '/'

dchi_comb_name1 = 'dchi_comb_' + type1 + '_' +'samples_' + str(starting_samp1) + '_' + str(starting_samp1+num_samps_tot-1) + dchi_comb_suffix
dchi_comb_name2 = 'dchi_comb_' + type2 + '_' +'samples_' + str(starting_samp2) + '_' + str(starting_samp2+num_samps_tot-1) + dchi_comb_suffix

if mode=='save':
    #run this cell to load dchi arrays with new file structure                                                                      
    #axis 0 : mask thickness, axis 1: mx or D0 pairing, axis 2: sigmav, axis 3: run number                                          
    #define dchi file names                                                                                               
    dchi_suffix = '_numsamps_' + str(num_samps_batch) + '_D0_' + str(D0) +'_mask_thicknesses_' + '{:.2e}'.format(mask_thicknesses[0]) +'_'+ '{:.2e}'.format(mask_thicknesses[-1]) + '_sigmav_'+ 'logspace_' + str(log_sigma_min) + '_' + str(log_sigma_max) + '_' + str(num_sigmav) + '_mx_logspace_'+str(mx_min) + '_' + str(mx_max) + '_' + str(num_mx) +'.npy'
    
    ss_batch1 = np.linspace(starting_samp1, starting_samp1+num_samps_tot-num_samps_batch, int(num_samps_tot/num_samps_batch))
    ss_batch2 = np.linspace(starting_samp2, starting_samp2+num_samps_tot-num_samps_batch, int(num_samps_tot/num_samps_batch))
    dchi_prename1 = ['dchi_' + type1 + '_starting_sample_'+ str(int(ss)) for ss in ss_batch1]
    dchi_prename2 = ['dchi_' + type2 + '_starting_sample_'+ str(int(ss)) for ss in ss_batch2]
    #initialize file name lists                                                                                                     

    print('file names sorted')
    
    test_array = np.load(dchi_path1 + dchi_prename1[0]+dchi_suffix)
    sh = test_array.shape
    num_masks = sh[0]
    num_mx = sh[1]
    num_sigmav = sh[2]

    #load arrays                                                                                                                    
    dchi1_tup = tuple(np.load(dchi_path1 + dchi_prename1[i]+dchi_suffix) for i in range(len(dchi_prename1)))
    dchi2_tup = tuple(np.load(dchi_path2 + dchi_prename2[i]+dchi_suffix) for i in range(len(dchi_prename2)))
    print('arrays loaded')

    #concatenate arrays                                                                                                             
    dchi1_comb = np.concatenate(dchi1_tup, axis=3)
    del dchi1_tup
    dchi2_comb = np.concatenate(dchi2_tup, axis=3)
    del dchi2_tup
    print('arrays have been concatenated')

    #save conbined arrays of dchi                                                                                                   
    np.save(dchi_comb_path + dchi_comb_name1, dchi1_comb)
    np.save(dchi_comb_path + dchi_comb_name2, dchi2_comb)
    print('combined dchi arrays have been saved')

elif mode=='load':
    #run this cell if dchi arrays have been stitched together                                                                       
    #load delta chi arrays                                                                                                          
    dchi1_comb = np.load(dchi_comb_path + dchi_comb_name1)
    dchi2_comb = np.load(dchi_comb_path + dchi_comb_name2)
    sh = dchi1_comb.shape
    num_masks = sh[0]
    num_mx = sh[1]
    num_sigmav = sh[2]
    print('dchi arrays have been loaded successfully')
else:
    print('mode is not recognized')

mean_dchi1 = np.mean(dchi1_comb, axis=3)
std_dchi1 = np.std(dchi1_comb, axis=3)
stdofmean_dchi1 = std_dchi1/np.sqrt(num_samps_tot)
mean_dchi2 = np.mean(dchi2_comb, axis=3)
std_dchi2 = np.std(dchi2_comb, axis=3)
stdofmean_dchi2 = std_dchi2/np.sqrt(num_samps_tot)

chisq_of_dchis = np.sum((mean_dchi1-mean_dchi2)**2/stdofmean_dchi2**2, axis=(1,2))
num_points = num_sigmav*num_mx
chisq_of_dchis_per = chisq_of_dchis/num_points

res_name = 'compare_dchi_ringvsnoring_'+dchi_suffix
stat_path = base_path + 'stat_results/'
np.save(stat_path + res_name, chisq_of_dchis_per)
