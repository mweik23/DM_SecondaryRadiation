#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:01:04 2021

@author: mitchellweikert
"""
import os
import numpy as np

def gen_fileinfo(out_index, mx=None, channel='bb_bar', sigma_v=None, Rmax=None, 
                 DA=None, D0=None, nu_range=None, rho_range=None, B_model=None, 
                 rho_star_model=None, DM_model=None, f_star_params=None, nr=None, 
                 nE = None, nrho=None, nnu=None):
    output_types = ['electron_spectrum', 'equillibrium_distribution', 'synchrotron_emission', 'inverse_compton_emission', 'photon_spectrum']
    content = output_types[out_index] + '\n\n' + \
    'Dark Matter Particle Mass: ' + str(mx) + 'GeV' + '\n' + \
    'Annihilation Channel: ' + str(channel) + '\n'
    if (not out_index == 0) and (not out_index==4):
        content = content + 'Thermally Averaged Cross-section: ' + str(sigma_v) + 'cm^3/s' + '\n' + \
        'Diffusion Radius: ' + str(Rmax) + 'kpc' + '\n' + \
        'Angular Distance: ' + str(DA) + 'kpc' + '\n'+ \
        'Diffusion Coefficient Normalization: ' + str(D0) + 'cm^2/s' + '\n'
        for model in B_model:
            content = content + 'Magnetic Field Model: ' + 'model type- '
            if model[0] == 'exp':
                content = content + 'Exponential; '
                content = content + 'B0- ' + str(model[1]) + 'uG; ' + 'Scale Radius- ' + str(model[2]) + 'kpc; \n'
            else:
                print('Magnetic Field Model Unidentified')
        for model in rho_star_model:
            content = content + 'Stellar Radiation Energy Density Model: ' + 'model type- '
            if model[0] == 'exp':
                content = content + 'Exponential; '
                content = content + 'rho0- ' + str(model[1]) + 'eV/cm^3; ' + 'Scale Radius- ' + str(model[2]) + 'kpc; \n'
            else:
                print('Stellar Radiation Model Unidentified')
        for model in DM_model:
            content = content + 'Dark matter Density Model: ' + 'model type- '
            if model[0] == 'exp':
                content = content + 'Exponential; '
                content = content + 'rho0- ' + str(model[1]) + 'GeV/cm^3; ' + 'Scale Radius- ' + str(model[2]) + 'kpc; \n'
            elif model[0] == 'nfw':
                content = content + 'NFW; '
                content = content + 'rho0- ' + str(model[1]) + 'GeV/cm^3; ' + 'gamma- '+ str(model[2]) + '; Scale Radius- ' + str(model[3]) + 'kpc; \n'
            else:
                print('Dark Matter Density Model Unidentified')
        content = content + 'Number of r steps: ' + str(nr) + '\n'
        content = content + 'Number of E steps: ' + str(nE) + '\n'
        if not (out_index == 1):
            content = content + 'Stellar Emmission Distribution Parameters: '  + 'average temperature- ' + str(f_star_params[0]) + 'K; ' + 'standard deviation of temperature- ' + str(f_star_params[1]) + 'K; \n'
            content = content + 'Number of rho steps: ' + str(nrho) + '\n'
            content = content + 'Number of nu steps: ' + str(nnu) + '\n'
            content = content + 'rho_range: ' + str(rho_range) + ' (kpc)' + '\n'
            content = content + 'nu_range: ' + str(nu_range) + ' (Hz)' + '\n'
    return content

def check_txt_files(file_info, path, temp=False):
    output_type = file_info.split('\n')[0]
    files = [f for f in os.listdir(path+output_type)]
    code = None
    output_name=None
    codes=[]
    for f in files:
        f_type = f.split('.')[-1]
        if f_type == 'txt' and not f[0]=='.':
            codes.append(f.split('_')[0])
            txt_file = open(path+output_type+ '/' + f, 'r')
            txt_file_read = txt_file.read()
            if txt_file_read == file_info:
                code = f.split('_')[0]
                break
    if code==None:
        exists = False
        check_code=True
        codes_int = [int(code) for code in codes]
        if not codes_int==[]:
            code = str(max(codes_int)+1).zfill(4)
        else:
            code = '0001'    

        file_info_name = path + output_type + '/' + code + '_' + output_type + '_info.txt'
        #create and save file info txt file
        f = open(file_info_name, "w")
        f.write(file_info)
        f.close()
        message = 'The ' + output_type + ' output does not exist. The newly generated file info .txt file, ' + file_info_name + ', was saved. The new output .npy file will be called: '
    else:
        exists = True
        file_info_name = path+output_type + '/' + code + '_' + output_type + '_info.txt'
        message = 'The ' + output_type + ' output does exist. The file info .txt file is located at: '  + file_info_name + '. The .npy output file is located at: '
    if temp:
        output_name = path + output_type + '/' + code + '_temporary_' + output_type + '.npy'
    else:
        output_name = path + output_type + '/' + code + '_' + output_type + '.npy'
    message = message + output_name
    #print(message)
    return output_name, file_info_name, exists, code

def find_results(out_type, mx=None, D0=None, DA=None, channel_in='bb_bar'):
    output_types = ['electron_spectrum', 'equillibrium_distribution', 'synchrotron_emission', 'inverse_compton_emission', 'photon_spectrum']
    this_path = os.path.realpath(__file__)
    print('this_path = ' + this_path)
    base_path = this_path.split('/scripts/')[0] + '/'
    #convert to sequence
    if np.isscalar(mx):
        mx = [mx]
    if np.isscalar(D0):
        D0 = [D0]
    if np.isscalar(DA):
        DA = [DA]
    if type(channel_in) == str:
        channel_in = [channel_in]
    
    if mx is not None:
        mx = [float(item) for item in mx]
    if D0 is not None:
        D0 = [float(item) for item in D0]
    if DA is not None:
        DA = [float(item) for item in DA]
    
    if type(out_type) == int:
        output_type = output_types[out_type] 
    elif out_type == 'espec' or 'electron_spectrum' or '0' or 0:
        output_type = output_types[0]
    elif out_type == 'equil' or 'equillibrium_distribution' or '1' or 1:
        output_type = output_types[1]
    elif out_type == 'sync' or 'synchrotron_emission' or '2' or 2:
        output_type = output_types[2]
    elif out_type == 'ic' or 'inverse_compton_emission' or '3' or 3:
        output_type = output_types[3]
    elif out_type == 'pspec' or 'photon_spectrum' or '4' or 4:
        output_type = output_types[4]
    else:
        print(str(out_type) + ' is not recognized as a possible value for the variable out_type')
        output_type=None
    
    file_list = [f for f in os.listdir(base_path+output_type) if f.split('.')[-1]=='txt']
    file_dict = []
    for f in file_list:
        txt_file = open(base_path + output_type + '/' + f, 'r')
        txt_file_read = txt_file.read()
        mx_str = txt_file_read.split('Dark Matter Particle Mass: ')[1].split('GeV')[0]
        channel_str = txt_file_read.split('Annihilation Channel: ')[1].split('\n')[0]
        if (not output_type == 'electron_spectrum') and (not output_type == 'photon_spectrum'):
           D0_str = txt_file_read.split('Diffusion Coefficient Normalization: ')[1].split('cm^2/s')[0]
        else:
            D0_str=None
        if output_type=='synchrotron_emission' or output_type=='inverse_compton_emission':
            DA_str = txt_file_read.split('Angular Distance: ')[1].split('kpc')[0]
        else:   
            DA_str=None
        inputs = [mx, D0, DA]
        strings = [mx_str, D0_str, DA_str]
        keep=True
        for i in range(len(inputs)):
            if not inputs[i]==None:
                if float(strings[i]) not in inputs[i]:
                    keep=False
        if channel_str not in channel_in:
            keep=False
        if keep:
            file_dict.append({'file_name' : f, 
                              'mx' : float(mx_str),
                              'D0' : float(D0_str),
                              'channel' : channel_str})
            if not (output_type=='photon_spectrum' or output_type=='electron_spectrum'):
                nr_str = txt_file_read.split('Number of r steps: ')[1].split('\n')[0]
                nE_str = txt_file_read.split('Number of E steps: ')[1].split('\n')[0]
                Rmax_str = txt_file_read.split('Diffusion Radius: ')[1].split('kpc')[0]
                sigmav_str = txt_file_read.split('Thermally Averaged Cross-section: ')[1].split('cm^3/s')[0]
                file_dict[-1]['nr'] = int(nr_str)
                file_dict[-1]['nE'] = int(nE_str)
                file_dict[-1]['sigma_v'] = float(sigmav_str)
                if not output_type=='equillibrium_distribution':
                    nrho_str = txt_file_read.split('Number of rho steps: ')[1].split('\n')[0]
                    nnu_str = txt_file_read.split('Number of nu steps: ')[1].split('\n')[0]
                    rho_range_str = txt_file_read.split('rho_range: (')[1].split(') ')[0]
                    rho_range_ls = rho_range_str.split(', ')
                    rho_range_ls_fl = [float(rho) for rho in rho_range_ls]
                    nu_range_str = txt_file_read.split('nu_range: (')[1].split(') ')[0]
                    nu_range_ls = nu_range_str.split(', ')
                    nu_range_ls_fl = [float(nu) for nu in nu_range_ls]
                    file_dict[-1]['DA'] = float(DA_str)
                    file_dict[-1]['nrho'] = int(nrho_str)
                    file_dict[-1]['nnu'] = int(nnu_str)
                    file_dict[-1]['rho_range'] = rho_range_ls_fl
                    file_dict[-1]['nu_range'] = nu_range_ls_fl
    return file_dict
       
if __name__=='__main__':
    file_info = gen_fileinfo(2, mx=50, channel='bb_bar', sigma_v=2.2e-26, Rmax=50, DA=780, D0=3e28, nu_range=[1e5, 1e22],
                rho_range=[100, 45000], B_model=[['exp', '10', '1.5'], ['exp', '8', '20']], rho_star_model=[['exp', '8', '4.3']], DM_model=[['nfw', '0.43', '1.25', '16.5']], f_star_params=[5500, 1000], nr=800, nE = 400, nrho=20, nnu=20)
    print(file_info)
    output_name, file_info_name, exists = check_txt_files(file_info)
    print('exists: ', exists)
