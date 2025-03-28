#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 10:47:24 2021

@author: mitchellweikert
"""

import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
import re
from scipy import interpolate
import manipulate_text as mt
import copy
import itertools
import argparse

#define relevant paths and names
script_path = os.path.realpath(__file__)
base_path =  script_path.split('scripts/')[0]
fits_path = base_path.split('Secondary_radiation')[0] + 'synchrotron_data/'
fits_name = 'm31cm3nthnew.ss.90sec.fits'

def get_data_and_info(fits_path, fits_name):
    #extract data and information about data
    hdul = fits.open(fits_path+fits_name)
    data_uJ = hdul[0].data[0]
    data = data_uJ/1000 #mJ/beam
    hdr = hdul[0].header
    dlt_N_deg = abs(hdr['CDELT1'])
    dlt_n_deg = abs(hdr['CDELT2'])
    N = hdr['NAXIS1']
    n = hdr['NAXIS2']
    nu_data = hdr['CRVAL3']
    nu_BW = hdr['CDELT3']
    HPBW_deg = hdr['BMIN']
    return data, dlt_N_deg, dlt_n_deg, N, n, HPBW_deg, nu_data, nu_BW 

def make_coords(N, n, dlt_N, dlt_n, loc='centered'):
    if loc=='centered':
        ax_N_unit = np.linspace(-(N-1)/2, (N-1)/2, N)
        ax_n_unit = np.linspace(-(n-1)/2, (n-1)/2, n)
    elif loc=='edges':
        ax_N_unit = np.linspace(-N/2, N/2, N+1)
        ax_n_unit = np.linspace(-n/2, n/2, n+1)
    return dlt_N*ax_N_unit, dlt_n*ax_n_unit

def gen_fake_data(beam_cut, frac, sigma_rms, sigma_BW, N, n, dlt_N, dlt_n):
    num_beams = beam_cut+1
    
    #create coords for data
    ax_N, ax_n = make_coords(N, n, dlt_N, dlt_n)
    AX_N, AX_n = np.meshgrid(ax_N, ax_n)
    
    #create coords for sources
    dlt_N_source = dlt_N/frac
    dlt_n_source = dlt_N_source
    fov_N = N*dlt_N
    print('fov N: ' + str(fov_N))
    fov_n = n*dlt_n
    print('fov n: ' + str(fov_n))
    fov_N_source = fov_N+2*num_beams*sigma_BW
    print('fov N source: ' + str(fov_N_source))
    fov_n_source = fov_n+2*num_beams*sigma_BW
    print('fov n source: ' + str(fov_n_source))
    N_source = int(np.round(fov_N_source/dlt_N_source))
    print('N source: ' + str(N_source))
    n_source = int(np.round(fov_n_source/dlt_n_source))
    print('n source: ' + str(n_source))
    fov_N_source = N_source*dlt_N_source
    fov_n_source = n_source*dlt_n_source
    ax_N_source, ax_n_source = make_coords(N_source, n_source, dlt_N_source, dlt_n_source)
    AX_N_SOURCE, AX_n_SOURCE = np.meshgrid(ax_N_source, ax_n_source)
    #smaller rms noise in the central lxb region where l and b are in degrees
    l = 2/3
    b = 2/3
    outside = np.logical_or(np.abs(AX_N_SOURCE)>(l/2)*np.pi/180, np.abs(AX_n_SOURCE)>(b/2)*np.pi/180)
    inside = np.logical_not(outside)
    noise = sigma_rms[-1]*outside + sigma_rms[0]*inside
    #generate fake data
    z = np.random.normal(size=(n_source, N_source))
    data_gen = 0*AX_N
    for i in range(n_source):
        for j in range(N_source):
            loc = (ax_n_source[i], ax_N_source[j])
            start_N = np.searchsorted(ax_N, loc[1]-beam_cut*sigma_BW, side='left')
            end_N = np.searchsorted(ax_N, loc[1]+beam_cut*sigma_BW, side='left')
            start_n = np.searchsorted(ax_n, loc[0]-beam_cut*sigma_BW, side='left')
            end_n = np.searchsorted(ax_n, loc[0]+beam_cut*sigma_BW, side='left')
            beam = dlt_N_source*noise[i,j]/(np.sqrt(np.pi)*sigma_BW)*z[i, j] \
                *np.exp(-((AX_N[start_n:end_n, start_N:end_N]-ax_N_source[j])**2+(AX_n[start_n:end_n, start_N:end_N]-ax_n_source[i])**2)/(2*sigma_BW**2))
            data_gen[start_n:end_n, start_N:end_N] += beam
    return data_gen, AX_N, AX_n

def gen_templates(run, nu_data, omega_beam, AX_N, AX_n):
    n, N = AX_N.shape
    #bring in variables from the run
    DA = run['DA']
    txt_file_name = run['file_name']
    nrho = run['nrho']
    nnu = run['nnu']
    sigmav_bench = run['sigma_v']
    rho_range = run['rho_range']
    nu_range = run['nu_range']
    
    #process run variables
    array_file_name = txt_file_name.split('_info')[0] + '.npy'
    m = re.search('[0-9]{4}_', array_file_name)
    out_type = array_file_name.split('.npy')[0].split(m.group(0))[-1]
    nu_v = np.logspace(np.log10(nu_range[0]), np.log10(nu_range[1]), nnu)
    rho_v = np.linspace(rho_range[0], rho_range[1], nrho)
    theta_v = np.arcsin(rho_v/np.sqrt(rho_v**2+DA**2))
    thetatheta, nunu = np.meshgrid(theta_v, nu_v)
    
    #load flux and change units
    nu_dSdth = np.load(base_path + out_type + '/' + array_file_name) #erg cm^-2 s^-1 rad^-1
    dS_dnudOmega = ((10**26/nunu)*nu_dSdth/(2*np.pi*np.sin(thetatheta))).astype(np.float64) #mJy/Sr
    rho_v_low = np.linspace(0, rho_range[0], 5)
    theta_v_low = np.arcsin(rho_v_low/np.sqrt(rho_v_low**2+DA**2))
    thetatheta_low, nunu_low = np.meshgrid(theta_v_low, nu_v)
    flux_low = 0*thetatheta_low
    
    #find max synch frequency index and extrapolate to rho=0
    delta_theta = (theta_v[1]-theta_v[0])/100
    domain = min(5,len(theta_v))
    for i in range(dS_dnudOmega.shape[0]):
        zs = np.where(dS_dnudOmega[i] == 0)[0]
        if len(zs) == nrho:
            nu_fin_ind = i #max frequency index + 1
            break
        else:
            for elem in zs:
                dS_dnudOmega[i, elem] = 1e-300
            oned_interp_nu = interpolate.interp1d(theta_v[0:domain], dS_dnudOmega[i][0:domain], kind = 'cubic')
            f_pr = (oned_interp_nu(theta_v[0]+delta_theta) - oned_interp_nu(theta_v[0]))/delta_theta
            f_prpr = f_pr/theta_v[0]
            f_0 = oned_interp_nu(theta_v[0])-f_prpr*theta_v[0]**2/2
            flux_low[i] = f_0+f_prpr*theta_v_low**2/2
    #combine low rho solution with main solution
    dS_dnudOmega_extrap = np.hstack((flux_low[:, :-1], dS_dnudOmega))
    theta_v_extrap = np.hstack((theta_v_low[:-1], theta_v))
    
    ln_dS_dnudOmega_extrap = np.log(dS_dnudOmega_extrap[0:nu_fin_ind])
    ln_dS_dnudOmega_interp = interpolate.interp2d(theta_v_extrap, nu_v[0:nu_fin_ind], ln_dS_dnudOmega_extrap[0:nu_fin_ind], kind='cubic')
    
    def dS_dnudOmega_cart(th_x, th_y, nu):
        return np.exp(ln_dS_dnudOmega_interp(np.sqrt(th_x**2+th_y**2), nu))
    
    #generate expected signal flux per beam in each pixel. The real way to do this is
    #to convolve real signal with beam using sig_meas function. This is slow and after testing
    #on a few points it seems that dS_dndOmega*Omega_beam is a good approximation.
    signal_beam = np.zeros((n,N)) 
    for i in range(signal_beam.shape[0]):
        for j in range(signal_beam.shape[1]):    
            signal_beam[i, j] = dS_dnudOmega_cart(AX_N[i, j], AX_n[i, j], nu_data)*omega_beam
    back_beam = np.ones((n, N))
    return signal_beam, back_beam

#coords = (x, y)
#ellipse_params = (r2, a, thickness/2)
def create_mask(THETA, th_range, coords=None, ellipse_params=None, data=None,  sigma_rms=None, num_sigma_rms=None, sigma_BW=None, num_sigma_BW=None, dlt=None, method='azimuthally_sym'):
    if method=='azimuthally_sym':
        return np.logical_and(THETA>th_range[0], THETA<th_range[1])
    elif method=='ellipse_and_azimuthally_sym':
        m1 = np.logical_and(THETA>th_range[0], THETA<th_range[1])
        elliptic_d = np.sqrt(coords[0]**2+ellipse_params[0]*coords[1]**2)
        m2 = np.abs(elliptic_d-ellipse_params[1])>ellipse_params[2]
        return m1*m2

def sort_runs(mx_set, D0_set, args=None, rtrn='inds'):
    D0_vals = np.sort(np.unique(D0_set))
    ind_set = []
    for D0_val in D0_vals:
        ind_D0val = np.where(D0_set==D0_val)[0]
        p_D0val = np.argsort(mx_set[ind_D0val])
        ind_set.append(ind_D0val[p_D0val])
    ind_set_tup = tuple(ind_set)
    ind_full = np.hstack(ind_set_tup)
    if rtrn == 'inds':
        return ind_full
    elif rtrn == 'arrs':
        if args==None:
            return mx_set[ind_full], D0_set[ind_full]
        else:
            return mx_set[ind_full], D0_set[ind_full], tuple(args[i][ind_full] for i in range(len(args)))
 
def f(x, w):
    elliptic_d = np.sqrt(x[0]**2+w[2]*x[1]**2)
    return w[0]+w[1]*np.exp(-(elliptic_d-w[3])**2/(2*w[4]))

def find_test_stats(resid, signal, background, sigma_rms, mask, this_ws, this_dw, this_chi_sb_sb, sigmav, sigmav_bench=2.2e-26):
    this_w0b = this_ws
    this_chi_b_b = this_chi_sb_sb
    this_w0s = this_ws+(sigmav/sigmav_bench)*this_dw
    this_wb = this_ws-(sigmav/sigmav_bench)*this_dw
    this_chi_b_sb = np.sum(mask*(resid+(sigmav/sigmav_bench)*signal-this_w0s*background)**2/(sigma_rms**2))
    this_chi_sb_b = np.sum(mask*(resid-(sigmav/sigmav_bench)*signal-this_wb*background)**2/(sigma_rms**2))
    this_delta_chi_b = this_chi_sb_b-this_chi_b_b
    this_delta_chi_sb = this_chi_sb_sb-this_chi_b_sb 
    return this_ws, this_w0b, this_w0s, this_wb, this_chi_sb_b, this_chi_b_sb, this_delta_chi_b, this_delta_chi_sb

if __name__=='__main__':
    
    #parse command line input
    parser = argparse.ArgumentParser()
    parser.add_argument('--ss', help='starting sample (the initial sample number of the run)', type=int)
    parser.add_argument('--mode', help='options are save, load, and none', type=str)
    parser.add_argument('--num_samples', help='this will be the number of samples generated in this script', type=int)
    parser.add_argument('--ellipse_params', help='these are the parameters of the ellipse of best fit ordered as [w_r, r^2, a, sigma_a^2]', nargs='+', type=float)
    args = parser.parse_args()
    
    #make them local variables
    mode = args.mode
    starting_sample = args.ss
    num_samples = args.num_samples
    
    #extract info about data
    data, dlt_N_deg, dlt_n_deg, N, n, HPBW_deg, nu_data, nu_BW = get_data_and_info(fits_path, fits_name)
    dlt_N = dlt_N_deg*np.pi/180
    dlt_n = dlt_n_deg*np.pi/180
    HPBW = HPBW_deg*np.pi/180
    sigma_BW = HPBW/(2*np.sqrt(2*np.log(2)))
    omega_beam = 2*np.pi*sigma_BW**2
    print('real data info extracted')
    
    
    #create coords for data
    ax_N, ax_n = make_coords(N, n, dlt_N, dlt_n)
    AX_N, AX_n = np.meshgrid(ax_N, ax_n)
    THETA = np.sqrt(AX_N**2+AX_n**2)
    
    #define noise charracterizing variables
    sigma_rms_vals = (0.25, 0.3)
    l = 2/3
    b = 2/3
    outside = np.logical_or(np.abs(AX_N)>(l/2)*np.pi/180, np.abs(AX_n)>(b/2)*np.pi/180) 
    inside = np.logical_not(outside)
    sigma_rms = sigma_rms_vals[0]*inside+sigma_rms_vals[-1]*outside
    
    #make masks
    th_range = [0.002, np.pi]
    num_masks = 10
    mask_set = []
    mask_thicknesses = np.linspace(16*np.sqrt(args.ellipse_params[3])/9, 28*np.sqrt(args.ellipse_params[3])/9, num_masks)
    for thickness in mask_thicknesses:
        mask = create_mask(THETA, th_range, coords=(AX_N, AX_n), ellipse_params=(args.ellipse_params[1], args.ellipse_params[2], thickness), method='ellipse_and_azimuthally_sym')
        mask_set.append(mask)
    mask_set = np.array(mask_set)
    #prepare sets of sigmav mx and D0
    frac = 2.7
    beam_cut = 5
    sig_type = 2
    mx_start = 6
    mx_stop = 500
    num_mx = 20
    sigmav_start = -27
    sigmav_end = -24
    num_sigmav = 80
    
    #make these be  in increasing order
    D0_set_in = np.array([3e28])
    mx_set_in = np.round(np.logspace(np.log10(mx_start), np.log10(mx_stop), num_mx), 1)
    sigmav_set = np.logspace(sigmav_start, sigmav_end, num_sigmav)
    
    #extract info of runs that match criteria
    run_list = mt.find_results(sig_type, mx=mx_set_in, D0=D0_set_in)
    num_runs = len(run_list)
    signal_temp_set = []
    mx_set_out = []
    D0_set_out = []
    for run in run_list:
        mx_set_out.append(run['mx'])
        D0_set_out.append(run['D0'])
        this_signal_temp, _ = gen_templates(run, nu_data, omega_beam, AX_N, AX_n)
        signal_temp_set.append(this_signal_temp)
    mx_set_out = np.array(mx_set_out)
    D0_set_out = np.array(D0_set_out)
    signal_temp_set = np.array(signal_temp_set)
    
    #sort runs
    ind = sort_runs(mx_set_out, D0_set_out)
    mx_set_out = mx_set_out[ind]
    D0_set_out = D0_set_out[ind]
    signal_temp_set = signal_temp_set[ind]
    run_list = [run_list[i] for i in ind]
    
    #initialize arrays
    ws = np.zeros((num_masks, num_samples))
    w0b = ws
    dw = np.zeros((num_masks, num_runs))
    chi_sb_sb = np.zeros((num_masks, num_samples))
    chi_b_sb = np.zeros((num_masks, num_runs, num_sigmav, num_samples))
    chi_sb_b = np.zeros((num_masks, num_runs, num_sigmav, num_samples))
    delta_chi_sb = np.zeros((num_masks, num_runs, num_sigmav, num_samples))
    delta_chi_b = np.zeros((num_masks, num_runs, num_sigmav, num_samples))
    chi_b_b = np.zeros((num_masks, num_samples))
    
    #this is a useful quantity and is independent of the run, sigmav and the generated image
    norm = np.sum(mask_set/(sigma_rms**2), axis=(1,2))
    
    #generate isotropic background template
    back_temp = np.ones((n, N))

    for i in range(num_samples):
            print('starting to generate sample number ' + str(i))
            #generate fake data residuals
            file_name = base_path + 'fake_resid_new/sigma_rms_'+ str(sigma_rms_vals[0]) + '_' + str(sigma_rms_vals[-1]) + '_spacing_' + '{:.2e}'.format(dlt_N/frac) + '_samplenum_' + str(i+starting_sample) + '.npy'
            if mode == 'save' or mode == 'none':
                data_resid, _, _ = gen_fake_data(beam_cut, frac, sigma_rms_vals, sigma_BW, N, n, dlt_N, dlt_n)
                if mode == 'save':
                    np.save(file_name, data_resid)
                print('sample number '+ str(i) + ' generated')
            elif mode == 'load':
                data_resid = np.load(file_name)
            else:
                print('value mode is not recognized')
            
            #add residuals to best fit ring model
            w_best = [0] + args.ellipse_params
            fake_data = data_resid + f([AX_N, AX_n], w_best)
# =============================================================================
#             if i == 0:
#                 plt.imshow(fake_data)
#                 plt.show()
#                 plt.close()
# =============================================================================
            for m in range(num_masks):
                ws[m, i] = np.sum(fake_data*mask_set[m]*back_temp/(sigma_rms**2))/norm[m]
                w0b[m, i] = ws[m, i]
                chi_sb_sb[m, i] =  np.sum(mask_set[m]*(fake_data-ws[m, i]*back_temp)**2/(sigma_rms**2)) 
                chi_b_b[m, i] = chi_sb_sb[m, i]
                
                for j in range(len(run_list)):
                    run = run_list[j]
                    if i==0:
                        dw[m, j] = np.sum(signal_temp_set[j]*mask_set[m]*back_temp/(sigma_rms**2))/norm[m]
                        
                    sigmav_bench = 2.2e-26
                    for l in range(len(sigmav_set)):
                        sigmav = sigmav_set[l]
                        this_ws, this_w0b, this_w0s, this_wb, this_chi_sb_b, this_chi_b_sb, this_delta_chi_b, this_delta_chi_sb \
                            =find_test_stats(fake_data, signal_temp_set[j], back_temp, sigma_rms, mask_set[m], ws[m, i], dw[m, j], chi_sb_sb[m, i], sigmav, sigmav_bench=2.2e-26)
                        chi_b_sb[m, j, l, i] = this_chi_b_sb
                        chi_sb_b[m, j, l, i] = this_chi_sb_b
                        delta_chi_b[m, j, l, i] = this_delta_chi_b
                        delta_chi_sb[m, j, l, i] = this_delta_chi_sb
                        
    dchi_suffix = 'starting_sample_' + str(starting_sample)+ '_numsamps_' + str(num_samples) + '_D0_' + str(D0_set_in[0]) +'_mask_thicknesses_' + '{:.2e}'.format(mask_thicknesses[0]) +'_'+ '{:.2e}'.format(mask_thicknesses[-1]) + '_sigmav_'+ 'logspace_' + str(sigmav_start) + '_' + str(sigmav_end) + '_' + str(num_sigmav) + '_mx_logspace_'+str(mx_start) + '_' + str(mx_stop) + '_' + str(num_mx) 
    file_name_sb = 'dchi_sb_'+ dchi_suffix + '.npy'
    file_name_b = 'dchi_b_'+ dchi_suffix + '.npy'
    np.save(base_path + 'dchi_masks/' + file_name_sb, delta_chi_sb)
    np.save(base_path + 'dchi_masks/' + file_name_b, delta_chi_b)
# =============================================================================
#     for m2 in range(5):
#         m = m2*2
#         for j2 in range(1):
#             j = j2
#             this_mx =mx_set_out[j]
#             this_D0 = D0_set_out[j]
#             for l2 in range(10):
#                 l = l2*8
#                 sigmav = sigmav_set[l]
#                 max_delta_chi = max(np.max(delta_chi_sb[m, j, l, :]), np.max(delta_chi_b[m, j, l, :]))
#                 min_delta_chi = min(np.min(delta_chi_sb[m, j, l, :]), np.min(delta_chi_b[m, j, l, :]))
#                 space = (max_delta_chi-min_delta_chi)/50
#                 max_bin = max_delta_chi + space
#                 min_bin= min_delta_chi - space
#                 num_bins = int(np.round(1.5*num_samples/10))
#                 #num_bins = 20
#                 bins = np.linspace(min_bin, max_bin, num_bins)
#                 #hist_suffix = 'starting_sample_' + str(starting_sample) + '_numsamps_' + str(num_samples) + '_mx_' + str(this_mx) + '_D0_' + str(this_D0) + '_sigmav_' + '{:.2e}'.format(sigmav)
#                 fig = plt.figure()
#                 plt.hist(delta_chi_sb[m, j, l, :], bins, label='s+b', alpha=0.5)
#                 plt.hist(delta_chi_b[m, j, l, :], bins, label='b', alpha=0.5)
#                 plt.xlabel('Delta chi^2')
#                 plt.ylabel('Frequency')
#                 plt.title('mx = ' + str(this_mx) + ' Gev' + '; D0 = ' + '{:.2e}'.format(this_D0) + ' cm^2/s'+ '; sigmav = ' + '{:.2e}'.format(sigmav) + ' cm^3/s; ' + 'mask_thickness = ' + '{:.2e}'.format(mask_thicknesses[m]))
#                 plt.legend()
# #                plt.savefig(base_path + '/dchi_hists/' + hist_suffix + '.pdf')
#                 plt.show()
#                 plt.close()
# =============================================================================
# ======================================================================
