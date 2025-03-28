#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:01:04 2021

@author: mitchellweikert
"""
import os
import numpy as np

def gen_fileinfo(out_index, mx=None, channel='bb_bar', sigma_v=None, rmax=None,\
                 D0=None, nu_range=None, nu_space=None, thx_range=None, thy_range=None, DM_model=None, \
                     astro_model=None, spherical_ave=None, nr=None, nE = None, nx=None, ny=None, nnu=None):
    output_types = ['electron_spectrum', 'equillibrium_distribution', 'synchrotron_emission', 'photon_spectrum']
    iesp = output_types.index('electron_spectrum')
    ipho = output_types.index('photon_spectrum')
    iequ = output_types.index('equillibrium_distribution')
    isyn = output_types.index('synchrotron_emission')
    content = output_types[out_index] + '\n\n' + \
    'Dark Matter Particle Mass: ' + str(mx) + 'GeV' + '\n' + \
    'Annihilation Channel: ' + str(channel) + '\n'
    if (not out_index == iesp) and (not out_index==ipho):
        content = content + 'Thermally Averaged Cross-section: ' + str(sigma_v) + 'cm^3/s' + '\n' + \
        'Diffusion Radius: ' + str(rmax) + 'kpc' + '\n' + \
        'Diffusion Coefficient Normalization: ' + str(D0) + 'cm^2/s' + '\n'
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
            
        content = content + 'Astrophysical Model: ' + astro_model + '\n'
        content = content + 'Spherical Average: ' + spherical_ave + '\n'
        content = content + 'Number of r steps: ' + str(nr) + '\n'
        content = content + 'Number of E steps: ' + str(nE) + '\n'
        if not (out_index == iequ):
            content = content + 'Number of x steps: ' + str(nx) + '\n'
            content = content + 'Number of y steps: ' + str(ny) + '\n'
            content = content + 'Number of nu steps: ' + str(nnu) + '\n'
            content = content + 'theta_x range: ' + str(thx_range) + ' rad' + '\n'
            content = content + 'theta_y range: ' + str(thy_range) + ' rad' + '\n'
            content = content + 'nu_range: ' + str(nu_range) + ' (Hz)' + '\n'
            content = content + 'nu_spacing: ' + nu_space + '\n'
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

def find_results(out_type, mx=None, D0=None, astro_model=None, spherical_ave=None, code=None, channel_in='bb_bar', DM_model=None):
    output_types = ['electron_spectrum', 'equillibrium_distribution', 'synchrotron_emission', 'photon_spectrum']
    this_path = os.path.realpath(__file__)
    #print('this_path = ' + this_path)
    base_path = this_path.split('/scripts/')[0] + '/'
    #convert to sequence
    if mx is not None:
        if np.isscalar(mx):
            mx = [mx]
        mx = [float(item) for item in mx]
    if D0 is not None:
        if np.isscalar(D0):
            D0 = [D0]
        D0 = [float(item) for item in D0]
    if type(channel_in) == str:
        channel_in = [channel_in]
    if type(code) == str:
        code = [code]
    if type(astro_model)==str:
        astro_model = [astro_model]
    if type(spherical_ave)==str:
        spherical_ave = [spherical_ave]
    if type(DM_model)== list:
        if np.isscalar(DM_model[0]) or type(DM_model[0]) == str:
            DM_model=[DM_model]
    
    if type(out_type) == int:
        output_type = output_types[out_type] 
    else:
        print(str(out_type) + ' is not recognized as a possible value for the variable out_type')
        output_type=None
    
    file_list = [f for f in os.listdir(base_path+output_type) if f.split('.')[-1]=='txt']
    file_dict = []
    for f in file_list:
        code_str = f.split('_')[0]
        txt_file = open(base_path + output_type + '/' + f, 'r')
        txt_file_read = txt_file.read()
        mx_str = txt_file_read.split('Dark Matter Particle Mass: ')[1].split('GeV')[0]
        channel_str = txt_file_read.split('Annihilation Channel: ')[1].split('\n')[0]
        if (not output_type == 'electron_spectrum') and (not output_type == 'photon_spectrum'):
            D0_str = txt_file_read.split('Diffusion Coefficient Normalization: ')[1].split('cm^2/s')[0]
            astro_model_str = txt_file_read.split('Astrophysical Model: ')[-1].split('\n')[0]
            sph_ave_str = txt_file_read.split('Spherical Average: ')[-1].split('\n')[0]
            DM_model_str = txt_file_read.split('Dark matter Density Model: ')[-1].split('\n')[0]
        else:
            D0_str=None
            astro_model_str = None
            sph_ave_str = None
            DM_model_str = None
        
        if DM_model_str is not None:
            DM_model_ls=[]
            DM_model_str_spl = DM_model_str.split('type- ')[-1].split(';')  
            DM_model_ls.append(DM_model_str_spl[0])
            DM_model_str_spl = DM_model_str_spl[-1].split('rho0- ')[-1].split('GeV')
            DM_model_ls.append(DM_model_str_spl[0])
            DM_model_str_spl = DM_model_str_spl[-1].split('gamma- ')[-1].split(';')
            DM_model_ls.append(DM_model_str_spl[0])
            DM_model_str_spl = DM_model_str_spl[-1].split('Scale Radius- ')[-1].split(';')
            DM_model_ls.append(DM_model_str_spl[0])
        
        
        inputs = [mx, D0]
        strings = [mx_str, D0_str]
        keep=True
        for i in range(len(inputs)):
            if not inputs[i]==None:
                if float(strings[i]) not in inputs[i]:
                    keep=False
        if channel_str not in channel_in:
            keep=False
        if type(code) == list:
            if code_str not in code:
                keep=False
        if type(astro_model)==list:
            if astro_model_str not in astro_model:
                keep=False
        if type(spherical_ave)==list:
            if sph_ave_str not in spherical_ave:
                keep=False
        if type(DM_model)==list:
            if DM_model_ls not in DM_model:
                keep=False
                
        if keep:
            file_dict.append({'file_name' : f, 
                              'mx' : float(mx_str),
                              'channel' : channel_str})
            if not (output_type=='photon_spectrum' or output_type=='electron_spectrum'):
                nr_str = txt_file_read.split('Number of r steps: ')[-1].split('\n')[0]
                nE_str = txt_file_read.split('Number of E steps: ')[-1].split('\n')[0]
                Rmax_str = txt_file_read.split('Diffusion Radius: ')[-1].split('kpc')[0]
                sigmav_str = txt_file_read.split('Thermally Averaged Cross-section: ')[-1].split('cm^3/s')[0]
                DM_model_str = txt_file_read.split('Dark matter Density Model: ')[-1].split('\n')[0]
                file_dict[-1]['D0'] = float(D0_str)
                file_dict[-1]['nr'] = int(nr_str)
                file_dict[-1]['nE'] = int(nE_str)
                file_dict[-1]['sigma_v'] = float(sigmav_str)
                file_dict[-1]['DM_model'] = DM_model_str
                file_dict[-1]['Astrophysical Model'] = astro_model_str
                file_dict[-1]['Spherical Average'] = sph_ave_str
                if not output_type=='equillibrium_distribution':
                    nx_str = txt_file_read.split('Number of x steps: ')[-1].split('\n')[0]
                    ny_str = txt_file_read.split('Number of y steps: ')[-1].split('\n')[0]
                    nnu_str = txt_file_read.split('Number of nu steps: ')[-1].split('\n')[0]
                    thx_range_str = txt_file_read.split('theta_x range: ')[-1].split(' rad')[0]
                    thy_range_str = txt_file_read.split('theta_y range: ')[-1].split(' rad')[0]
                    thx_range_fl = float(thx_range_str)
                    thy_range_fl = float(thy_range_str)
                    nu_range_str = txt_file_read.split('nu_range: (')[-1].split(') ')[0]
                    nu_range_ls = nu_range_str.split(', ')
                    nu_range_ls_fl = [float(nu) for nu in nu_range_ls]
                    nu_space_str = txt_file_read.split('nu_spacing: ')[-1].split('\n')[0]
                    file_dict[-1]['nx'] = int(nx_str)
                    file_dict[-1]['ny'] = int(ny_str)
                    file_dict[-1]['nnu'] = int(nnu_str)
                    file_dict[-1]['thx_range'] = thx_range_fl
                    file_dict[-1]['thy_range'] = thy_range_fl
                    file_dict[-1]['nu_range'] = nu_range_ls_fl
                    file_dict[-1]['nu_spacing'] = nu_space_str
    return file_dict
       
if __name__=='__main__':
    mx_set = np.round(np.logspace(np.log10(6), np.log10(500), 20), 1)
    D0_set = np.array([5e27, 1.1e28, 5e28, 2e29])
    print('mx: ', mx_set)
    print('D0: ', D0_set)

    run_list = find_results(2, astro_model='Andromeda5', spherical_ave='unweighted',  mx=mx_set, D0=D0_set)
    print(len(run_list))
