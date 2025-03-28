#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 12:21:36 2021

@author: mitchellweikert
"""

import numpy as np
from astropy.io import fits
import os
import matplotlib.pyplot as plt
import re
from scipy import interpolate
import manipulate_text as mt
#import stat_method_masking_cor_mat as sm
import copy
import itertools
import argparse


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
            beam = dlt_N_source*sigma_rms/(np.sqrt(np.pi)*sigma_BW)*z[i][j] \
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

def find_center(THETA, th_range):
    ind = np.where(np.logical_or(THETA<th_range[0], THETA>th_range[1]))
    return ind

#coords = (x, y)
#ellipse_params = (r2, a, thickness/2)
def where_to_mask(THETA, th_range, coords=None, ellipse_params=None, data=None,  sigma_rms=None, num_sigma_rms=None, sigma_BW=None, num_sigma_BW=None, dlt=None, method='azimuthally_sym'):
    if method=='thresh':
        min_data = np.min(data)
        thresh = min_data + 2*num_sigma_rms*sigma_rms
        ind_bri = np.where(data>thresh)
        ind_ran = np.where(np.logical_or(THETA<th_range[0], THETA>th_range[1]))
        ind_smear = []
        mask_range = int(np.round(num_sigma_BW*sigma_BW/dlt))
        for row, col in zip(ind_bri[0], ind_bri[1]):
            ind_smear_add = [[row+i, col+j] for i,j in itertools.product(range(-mask_range, mask_range+1), range(-mask_range, mask_range+1)) \
                         if row+i >= 0 and col+j >=0 and row+i < data.shape[0] and col+j < data.shape[1] and i**2+j**2 <= mask_range**2]
            ind_smear += ind_smear_add
        ind_bri_mask_t = np.unique(ind_smear, axis=0)
        ind_bri_mask = np.transpose(ind_bri_mask_t)
        if len(ind_ran[0])>0:
            ind_mask_raw = np.hstack((ind_bri_mask, np.array([ind_ran[0], ind_ran[1]])))
            ind_mask_arr = np.unique(ind_mask_raw, axis=1)
        else:
            ind_mask_arr = ind_bri_mask
        if len(ind_mask_arr.shape)==1:
            ind_mask_tup = tuple((np.array([]), np.array([])))
        else:
            ind_mask_tup = tuple((ind_mask_arr[0], ind_mask_arr[1]))
        return ind_mask_tup
    elif method=='azimuthally_sym':
        return find_center(THETA, th_range)
    elif method=='ellipse_and_azimuthally_sym':
        ind1 = find_center(THETA, th_range)
        elliptic_d = np.sqrt(coords[0]**2+ellipse_params[0]*coords[1])
        ind2 = np.where(np.abs(elliptic_d-ellipse_params[1])<ellipse_params[2])
        ind1_ar = np.array(ind1)
        ind2_ar = np.array(ind2)
        ind_ar = np.hstack((ind1_ar, ind2_ar))
        ind_ar_un = np.unique(ind_ar, axis=1)
        return ind_ar_un
        
#puts indices in decending order with respect to a certain axis
def sort_ind(ind, axis=0):
    p = ind[0].argsort()[::-1]
    new_ind = ()
    for elem in ind:
        new_ind += (elem[p],)
    return new_ind

def mask_image(image, ind , sort=True):
    sh = image.shape
    #reshape image if neccessary 
    if not len(sh) == 3:
        n = sh[0]
        N = sh[1]
        block_image = np.transpose(image).reshape(N,n,1)
    else:
        block_image = image
     
    #convert block_image to a list of arrays
    block_image_ls = list(block_image)
    
    #sort indices if neccesary
    if sort:
        #reorder ind
        ind = sort_ind(ind)
    
    #remove entries
    for i in range(len(ind[0])):
        block_image_ls[ind[1][i]] = np.delete(block_image_ls[ind[1][i]], ind[0][i], 0)
    
    return block_image_ls

def find_test_stats(resid, signal, background, sigma_rms, this_ws, this_dw, this_chi_sb_sb, sigmav, sigmav_bench=2.2e-26):
    this_w0b = this_ws
    this_chi_b_b = this_chi_sb_sb
    this_w0s = this_ws+(sigmav/sigmav_bench)*this_dw
    this_wb = this_ws-(sigmav/sigmav_bench)*this_dw
    this_chi_b_sb = sum([np.sum((resid[k]+(sigmav/sigmav_bench)*signal[k]-this_w0s*background[k])**2/(sigma_rms**2)) for k in range(len(resid))])
    this_chi_sb_b = sum([np.sum((resid[k]-(sigmav/sigmav_bench)*signal[k]-this_wb*background[k])**2/(sigma_rms**2)) for k in range(len(resid))])
    this_delta_chi_b = this_chi_sb_b-this_chi_b_b
    this_delta_chi_sb = this_chi_sb_sb-this_chi_b_sb 
    return this_ws, this_w0b, this_w0s, this_wb, this_chi_sb_b, this_chi_b_sb, this_delta_chi_b, this_delta_chi_sb

def produce_radial_plot(data_ma, signal_beam_ma, sigmav, mx, D0, this_w0b, this_wb, THETA_ma, dlt_n, dlt_N, th_max, sigma_BW, sigmav_bench=2.2e-26):
    # produce course-grained plot to check stats
    N = len(data_ma)
    n_bins = int(np.floor(th_max/(2*sigma_BW)))
    th_bins = np.linspace(0, n_bins*2*sigma_BW, n_bins+1)
    bin_num = [np.digitize(THETA_ma[i], th_bins)-1 for i in range(N)]
    Omega_pix = dlt_n*dlt_N
    omega_beam = 2*np.pi*sigma_BW**2
    pix_per_beam = omega_beam/Omega_pix
    pix_per_ring = np.zeros(n_bins+1)
    flux_ring_sum = np.zeros(n_bins+1)
    sig_ring_sum = np.zeros(n_bins+1)
    for i in range(N):
        for j in range(data_ma[i].shape[0]):
            flux_ring_sum[bin_num[i][j,0]] += data_ma[i][j,0]
            sig_ring_sum[bin_num[i][j,0]] += (sigmav/sigmav_bench)*signal_beam_ma[i][j,0] 
            pix_per_ring[bin_num[i][j,0]] += 1
    flux_ring = flux_ring_sum/pix_per_beam  
    sig_ring = sig_ring_sum/pix_per_beam
    bg_ring = this_wb*pix_per_ring/pix_per_beam
    bg_only_ring = this_w0b*pix_per_ring/pix_per_beam
    err_ring = sigma_rms*np.sqrt(pix_per_ring/pix_per_beam)
    th_bin_cent = np.array([th_bins[i]+th_bins[i+1] for i in range(n_bins)])
    fig = plt.figure()
    plt.errorbar(th_bin_cent, flux_ring[:-1], yerr=err_ring[:-1], ls='none', label='data', color='r')
    plt.scatter(th_bin_cent, sig_ring[:-1] + bg_ring[:-1], label='s+b', color = 'b')
    plt.scatter(th_bin_cent, bg_only_ring[:-1], label='b', color= 'g')
    plt.title('mx = ' + str(mx) + ' sigmav = ' + "{:.2e}".format(sigmav) + ' D0 = ' + '{:.2e}'.format(D0))
    plt.xlabel('theta (rad)')
    plt.ylabel('flux (mJy/ring)')
    plt.legend()
    plt.savefig(base_path+'/radial_binned_plots/radial_binned_compare_fake_residuals_mx_' + str(mx) + '_sigmav_' + "{:.2e}".format(sigmav) + '_D0_' + "{:.1e}".format(D0) + '.pdf')
    plt.close()

if __name__=='__main__':
    #define relevant paths and names
    script_path = os.path.realpath(__file__)
    base_path =  script_path.split('scripts/')[0]
    fits_path = base_path.split('Secondary_radiation')[0] + 'synchrotron_data/'
    fits_name = 'm31cm3nthnew.ss.90sec.fits'
    
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
    
    #setting = 'compare to theory'
    setting = 'CLS'
    
    if setting=='CLS':
        
        #parse command line input
        parser = argparse.ArgumentParser()
        parser.add_argument('--ss', help='starting sample (the initial sample number of the run)', type=int)
        parser.add_argument('--mode', help='options are save, load, and none', type=str)
        parser.add_argument('--num_samples', help='this will be the number of samples generated in this script', type=int)
        parser.add_argument('--ellipse_params', help='these are the parameters of the ellipse of best fit ordered as [w_r, r^2, a, sigma_a^2]', nargs='+', type=float)
        args = parser.parse_args()
        
        mode = args.mode
        starting_sample = args.ss
        num_samples = args.num_samples

        #decide where to mask images
        th_range = [0, np.pi]
        sigma_rms = 0.25
        num_masks = 20
        mask_ind_sets = []
        mask_thicknesses = np.linspace(0, 4*np.sqrt(ellipse_params[3]), num_masks)
        for thickness in mask_thicknesses:
            mask_ind = where_to_mask(THETA, th_range, coords=(AX_N, AX_n), ellipse_params=(args.ellipse_params[1], args.ellipse_params[2], thickness))
            mask_ind = sort_ind(mask_ind)
            mask_ind_sets.append(mask_ind)

        #prepare for loops over samples
        frac = 3.2
        beam_cut = 5
        sig_type = 2
        mx_ls = list(np.round(np.logspace(np.log10(6), np.log10(500), 20), 1))
        sigmav_start = -28
        sigmav_end = -25
        sigmav_num = 80
        sigmav_set = np.logspace(sigmav_start, sigmav_end, sigmav_num)
        run_list = mt.find_results(sig_type, mx=mx_ls)
        p=[]
        ws = np.zeros(num_samples)
        w0b = ws
        norm = (1/sigma_rms**2)*(N*n-len(mask_ind[0]))
        dw = np.zeros(len(run_list))
        chi_sb_sb = np.zeros((num_masks, num_samples))
        chi_b_sb = np.zeros((num_masks, len(run_list), num_samples, len(sigmav_set)))
        chi_sb_b = np.zeros((num_masks, len(run_list), num_samples, len(sigmav_set)))
        delta_chi_sb = np.zeros((num_masks, len(run_list), num_samples, len(sigmav_set)))
        delta_chi_b = np.zeros((num_masks, len(run_list), num_samples, len(sigmav_set)))
        chi_b_b = np.zeros((num_masks, num_samples))
        back_temp = np.ones((n, N))
        back_temp = mask_image(back_temp, mask_ind, sort=False)
        if len(run_list)==1:
            signal_temp, _ = gen_templates(run_list[0], nu_data, omega_beam, AX_N, AX_n)
            signal_temp = mask_image(signal_temp, mask_ind, sort=False)
            
        for i in range(num_samples):
            print('starting to generate sample number ' + str(i))
            #generate fake data residuals
            file_name = base_path + 'fake_residuals/sigma_rms_'+ str(sigma_rms) + '_spacing_' + '{:.2e}'.format(dlt_N/frac) + '_samplenum_' + str(i+starting_sample) + '.npy'
            if mode == 'save' or mode == 'none':
                data_resid, _, _ = gen_fake_data(beam_cut, frac, sigma_rms, sigma_BW, N, n, dlt_N, dlt_n)
                if mode == 'save':
                    np.save(file_name, data_resid)
            elif mode == 'load':
                data_resid = np.load(file_name)
            else:
                print('value mode is not recognized')
            
            data_resid = mask_image(data_resid, mask_ind, sort=False)
            ws[i] = sum([np.sum(data_resid[k]*back_temp[k])/(sigma_rms**2) for k in range(len(data_resid))])/norm
            w0b[i] = ws[i]
            chi_sb_sb[i] =  sum([np.sum((data_resid[k]-ws[i]*back_temp[k])**2)/(sigma_rms**2) for k in range(len(data_resid))])
            chi_b_b[i] = chi_sb_sb[i]
            for j in range(len(run_list)):
                run = run_list[j]
                this_mx = run['mx']
                this_D0 = run['D0']
                #generate templates
                if not len(run_list)==1:
                    signal_temp, _ = gen_templates(run, nu_data, omega_beam, AX_N, AX_n)
                    signal_temp = mask_image(signal_temp, mask_ind, sort=False)
                if i==0:
                    dw[j] = sum([np.sum(signal_temp[k]*back_temp[k])/(sigma_rms**2) for k in range(len(signal_temp))])/norm
                
                sigmav_bench = 2.2e-26
                for l in range(len(sigmav_set)):
                    sigmav = sigmav_set[l]
                    this_ws, this_w0b, this_w0s, this_wb, this_chi_sb_b, this_chi_b_sb, this_delta_chi_b, this_delta_chi_sb \
                        =find_test_stats(data_resid, signal_temp, back_temp, sigma_rms, ws[i], dw[j], chi_sb_sb[i], sigmav, sigmav_bench=2.2e-26)
                    chi_b_sb[j, i, l] = this_chi_b_sb
                    chi_sb_b[j, i, l] = this_chi_sb_b
                    delta_chi_b[j, i, l] = this_delta_chi_b
                    delta_chi_sb[j, i, l] = this_delta_chi_sb
                    THETA_ma = mask_image(THETA, mask_ind, sort=False)
                    th_max = ax_n[-1]
#                    if i==0 and (l-2)%5 == 0:
#                        produce_radial_plot(data_resid, signal_temp, sigmav, this_mx, this_D0, this_w0b, this_wb, THETA_ma, dlt_n, dlt_N, th_max, sigma_BW, sigmav_bench=sigmav_bench)
        for j in range(len(run_list)):
            run = run_list[j]
            this_mx = run['mx']
            this_D0 = run['D0']
            dchi_suffix = 'starting_sample_' + str(starting_sample)+ '_numsamps_' + str(num_samples) + '_mx_' + str(this_mx) + '_D0_' + str(this_D0) + '_sigmav_'+ 'logspace_' + str(sigmav_start) + '_' + str(sigmav_end) + '_' + str(sigmav_num) 
            file_name_sb = 'dchi_sb_'+ dchi_suffix + '.npy'
            file_name_b = 'dchi_b_'+ dchi_suffix + '.npy'
            np.save(base_path + 'dchi/' + file_name_sb, delta_chi_sb[j])
            np.save(base_path + 'dchi/' + file_name_b, delta_chi_b[j])

#            for l in range(len(sigmav_set)):
#                sigmav = sigmav_set[l]
#                max_delta_chi = max(np.max(delta_chi_sb[j, :, l]), np.max(delta_chi_b[j, :, l]))
#                min_delta_chi = min(np.min(delta_chi_sb[j, :, l]), np.min(delta_chi_b[j, :, l]))
#                space = (max_delta_chi-min_delta_chi)/50
#                max_bin = max_delta_chi + space
#                min_bin= min_delta_chi - space
#                num_bins = int(np.round(1.5*num_samples/10))
#                #num_bins = 20
#                bins = np.linspace(min_bin, max_bin, num_bins)
#                hist_suffix = 'starting_sample_' + str(starting_sample) + '_numsamps_' + str(num_samples) + '_mx_' + str(this_mx) + '_D0_' + str(this_D0) + '_sigmav_' + '{:.2e}'.format(sigmav)
#                fig = plt.figure()
#                plt.hist(delta_chi_sb[j, :, l], bins, label='s+b', alpha=0.5)
#                plt.hist(delta_chi_b[j, :, l], bins, label='b', alpha=0.5)
#                plt.xlabel('Delta chi^2')
#                plt.ylabel('Frequency')
#                plt.title('mx = ' + str(this_mx) + ' Gev' + '; D0 = ' + '{:.2e}'.format(this_D0) + ' cm^2/s'+ '; sigmav = ' + '{:.2e}'.format(sigmav) + ' cm^3/s')
#                plt.legend()
#                plt.savefig(base_path + '/dchi_hists/' + hist_suffix + '.pdf')
#                plt.close()
# =============================================================================
#     elif setting=='compare to theory':
#         #generate correlation matrix
#         sigma_rms = 0.25
#         block_rows = sm.construct_mat(sigma_rms, sigma_BW, N, n, dlt_N, dlt_n, cut_thresh=0.2)
#         block_rows_ls = [list(block_rows[i]) for i in range(N)]
#         
#         sig_type = 2
#         mx = 15.2
#         run_list = mt.find_results(sig_type, mx)
#         p=[]
#         w=[]
#         w0=[]
#         chi_sb=[]
#         chi_b=[]
#         
#         for run in run_list:
#             #generate templates
#             signal_temp, back_temp = gen_templates(run, nu_data, omega_beam, AX_N, AX_n)
#             sigmav_bench = 2.2e-26
#             sigmav = 3e-25
#             signal_temp_ls = list(np.transpose(signal_temp).reshape(N,n,1))
#             backgr_temp_ls = list(np.transpose(back_temp).reshape(N,n,1))
#             
#             #prepare for loop over samples
#             num_samples = 10
#             p_samp = []
#             w_samp = []
#             w0_samp = []
#             chi_sb_samp = []
#             chi_b_samp = []
#             for i in range(num_samples):
#                 #generate fake data residuals
#                 frac = 1.2
#                 beam_cut = 6
#                 data_resid, _, _ = gen_fake_data(beam_cut, frac, sigma_rms, sigma_BW, N, n, dlt_N, dlt_n)
#                 print('fake data residuals generated')
#         
#                 #plot the residuals
#                 plt.imshow(data_resid, extent=[AX_N[0, 0], AX_N[-1, -1], AX_n[-1, -1], AX_n[0, 0]])
#                 plt.title('fake data residuals (mJy/beam)')
#                 plt.xlabel('angle from center (rad)')
#                 plt.ylabel('angle from center (rad)')
#                 plt.colorbar(orientation='horizontal')
#                 #plt.savefig(base_path+'/figs/fake_data_residuals.pdf')
#                 plt.show()
#                 
#                 #create data with signal 
#                 data_with_signal = (sigmav/sigmav_bench)*signal_temp+data_resid
#                 data_with_signal = list(np.transpose(data_with_signal).reshape(N,n,1))
#                 
#                 #get statistics for this data
#                 this_p, this_w, this_w0, this_chi_b, this_chi_sb = sm.find_pvalue(data_with_signal, signal_temp_ls, backgr_temp_ls, sigmav, block_rows_ls, sigmav_bench=sigmav_bench)
#                 p_samp.append(this_p)
#                 w_samp.append(this_w)
#                 w0_samp.append(this_w0)
#                 chi_sb_samp.append(this_chi_sb)
#                 chi_b_samp.append(this_chi_b)
#             p.append(p_samp)
#             w.append(w_samp)
#             w0.append(w0_samp)
#             chi_b.append(chi_b_samp)
#             chi_sb.append(chi_sb_samp)
# =============================================================================
    
