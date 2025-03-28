#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 14:25:18 2021

@author: mitchellweikert
"""
print('compute signal has begun')
import numpy as np
from scipy.integrate import quad
import meta_variables as mv
import constants_and_functions as cf
import argparse
import os
import manipulate_text as mt
import get_espec as ge
import matplotlib.pyplot as plt
import proceedures as pr
import time
print('packages have been imported')
path_name = os.path.realpath(__file__).split('scripts')[0]
output_type = ['electron_spectrum', 'equillibrium_distribution', 'synchrotron_emission', \
               'inverse_compton_emission', 'photon_spectrum']
print('path and output types have been set')

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.lower() == 'none':
        return None
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def main(mx, sigma_v, Rmax, DA, D0, nu_range, rho_range, B_model, rho_star_model, \
         DM_model, f_star_params, output_names, file_info_names, exists, codes,\
             overwrite=5*[False], temp=5*[False], channel='bb_bar', E_spacing='log'):
    print('main function has started running')
    equil = True
    sync = True
    ic = True
    if output_names[1] is None:
        equil=False
    if output_names[2] is None:
        sync = False
    if output_names[3] is None:
        ic=False
        
    #construct espec bins name
    espec_name_split = output_names[0].split('/')
    
    if temp[0]:
        espec_name_split[-1] = codes[0] + '_temporary_' + espec_name_split[-2] + '_bins.npy'
        espec_fig_name = codes[0] + '_temporary_' + output_type[0] + '.pdf'
    else:
        espec_name_split[-1] = codes[0] + '_' + espec_name_split[-2] + '_bins.npy'
        espec_fig_name = codes[0] + '_' + output_type[0] + '.pdf'
    espec_bins_name = ''
    for part in espec_name_split:
        if not (part ==''):
            espec_bins_name += '/' + part
    #constuct path for bins array
    gammaspec_name_split = output_names[4].split('/')
    code = gammaspec_name_split[-1].split('_')[0]
    if temp[4]:
        gammaspec_name_split[-1] = codes[4] + '_temporary_' + gammaspec_name_split[-2] + '_bins.npy'
        gammaspec_fig_name = codes[4] + '_temporary_' + output_type[4] + '.pdf'
    else:
        gammaspec_name_split[-1] = codes[4] + '_' + gammaspec_name_split[-2] + '_bins.npy'
        gammaspec_fig_name = codes[4] + '_' + output_type[4] + '.pdf'
    gammaspec_bins_name = ''
    for part in gammaspec_name_split:
        if not (part==''):
            gammaspec_bins_name += '/' + part
    
    # If both electron spectrum and photon spectrum exist and I do not want to overwrite them, 
    #then I don't have to process the hepmc files and can just load the electron spectrum and photon spectrum
    if exists[0] and exists[4] and not overwrite[0] and not overwrite[4] and not temp[0] and not temp[4]:
        print('MESSAGE: Electron spectrum and photon spectrum already exists for these parameters.' + \
              ' Loading electron spectrum from: ' + output_names[0] + '. Loading photon spectrum from ' + output_names[4])
        e_spec = np.load(output_names[0])
        e_bins = np.load(espec_bins_name)
        gamma_spec = np.load(output_names[4])
        gamma_bins = np.load(gammaspec_bins_name)
    # otherwise, I will process the hepmc files
    
    else:
        print('MESSAGE: Either electron or photon spectrum does not exist. Searching for hepmc file(s)...')
        hepmc_paths = ge.get_hepmc_names(mx)
        # if there are no hepmc files with the correct energy then the program cannot move on               
        if hepmc_paths==[]:
            print('MESSAGE: no hepmc files for the requested Dark Matter mass. '+ \
                  'Use pythia to generate E_beam = ' + str(mx) + ' GeV')
        # as long as there is at least one hepmc file with the correct energy, I will extract e and gamma spectra          
        else:
            print('MESSAGE: ' + str(len(hepmc_paths))+ ' hepmc file(s) exist(s). Converting it/them to a electron and photon spectra...')
            eE = []
            gammaE = []
            numev = 0
            for hepmc_path in hepmc_paths:
                eE_cur, gammaE_cur, numev_cur = ge.hepmc_to_list(hepmc_path)
                eE = eE + eE_cur
                gammaE = gammaE + gammaE_cur
                numev = numev + numev_cur

            e_bins, e_spec, gamma_bins, gamma_spec = ge.get_spectra(eE, gammaE, numev, mx)
            # if the photon spectrum has not been extracted, then I will create the proper bins path name and save a .txt file for the photon spectrum
            if not (exists[4] and not overwrite[4]) or temp[4]:   
                print('MESSAGE: Photon spectrum does not exist. Saving photon spectrum and bins')
                
                #save gamma spectrum and bins
                np.save(output_names[4], gamma_spec)
                np.save(gammaspec_bins_name, gamma_bins)
                print('MESSAGE: Photon spectrum saved at: ', output_names[4])
                print('MESSAGE: Photon spectrum bins saved at: ', gammaspec_bins_name)
                #make and save plot of electron spectrum
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.plot(gamma_bins[:-1], gamma_spec)
                ax.set_xscale('log')
                ax.set_yscale('log')
                plt.xlabel('E (GeV)')
                plt.ylabel('dN_gamma/dE (GeV^-1)')
                plt.title('m_x = '+ str(mx) + ' GeV')
                plt.savefig(path_name + output_type[4] + '/' + gammaspec_fig_name)
                
                
                print('MESSAGE: Photon spectrum plot saved at: '+ path_name + output_type[4] + '/' + gammaspec_fig_name)
            else:
                print('MESSAGE: Photon spectrum exists. Loading spectrum and bins from ' + output_names[4] + ' and ' + gammaspec_bins_name + ', respectively.')
                gamma_spec = np.load(output_names[4])
                gamma_bins = np.load(gammaspec_bins_name)
            #if electron spectrum does not exist
            if not (exists[0] and not overwrite[0]) or temp[0]:
                print('MESSAGE: Electron spectrum does not exist. Saving electton spectrum and bins')
                #save e spectrum and bins
                np.save(output_names[0], e_spec)
                np.save(espec_bins_name, e_bins)
                print('MESSAGE: Electron spectrum saved at: ', output_names[0])
                print('MESSAGE: Electron spectrum bins saved at: ', espec_bins_name)
                #make and save plot of electron spectrum
                fig = plt.figure()
                ax = fig.add_subplot()
                ax.plot(e_bins[1:], e_spec)
                ax.set_xscale('log')
                ax.set_yscale('log')
                plt.xlabel('E (GeV)')
                plt.ylabel('dN_e/dE (GeV^-1)')
                plt.title('m_x = '+ str(mx) + ' GeV')
                plt.savefig(path_name + output_type[0]+ '/' + espec_fig_name)
                
                print('MESSAGE: Electron spectrum plot saved at: '+ path_name + output_type[0] + '/' + espec_fig_name)
            else:
                print('MESSAGE: Electron spectrum exists. Loading spectrum and bins from ' + output_names[0] + ' and ' + espec_bins_name + ', respectively.')
                e_spec = np.load(output_names[0])
                e_bins = np.load(espec_bins_name)
    
    #if exists[2] and not overwrite[2] and exists[3] and not overwrite[3]:
        #message = 'Synchrotron and Inverse Compton flux distributions for these parameters already exist at: '
        #message = message + output_names[2] + ' and' + output_names[3] + ', respectively'
        #print(message)
    if equil:
        B_r, dBdr_r, rho_star_r, rho_DM_r = pr.gen_radial_funcs(B_model, rho_star_model, DM_model) 
        #if equillibrium distribution result already exist and I do not want to overwrite it, then load  equillibrium dist
        if exists[1] and not overwrite[1] and not temp[1]:
            print('MESSAGE: Equillibrium distribution for these variables exists at: ', output_names[1])
            u = np.load(output_names[1])
            #check if shape is correct
            if u.shape == (mv.nE, mv.nr-1):
                print('MESSAGE: Equillibrium phase-space distribution successfully loaded')
                rr, EE, dr, dE = mv.grid(mv.nr, mv.nE, [0, Rmax], [cf.me, mx], E_spacing=E_spacing)
            else:
                statement = 'MESSAGE: The shape of the loaded equillibrium phase-space distribution does not match grid shape from meta variables.'
                statement = statement + ' Change the grid-size in meta_variables.py so that it matches the loaded file, or overwrite the existing electron equillibrium distribution.'
                print(statement)
        #otherwise compute equillibrium distribution
        else:
            rr, EE, u = pr.find_equillibrium(mx, sigma_v, D0, Rmax, e_spec, e_bins, funcs=[B_r, dBdr_r, rho_star_r, rho_DM_r], E_spacing=E_spacing)
            np.save(output_names[1], u)
            print('MESSAGE: Equillibrium distribution computed and saved at: ' + output_names[1])
            
            if temp[1]:
                equildist_figname = codes[1] + '_temporary_' + output_type[1] + '.pdf'
            else:
                equildist_figname = codes[1] + '_' + output_type[1] + '.pdf'
            fig = plt.figure()
            ax = fig.add_subplot()
            colors = ['m', 'r', 'orange', 'g', 'c', 'b', 'k']
            for i in range(7):
                step_E = np.floor(mv.nE/7)-10
                ind_E = int((i+1)*step_E)
                ax.plot(rr[ind_E], ((cf.cm_per_pc)**(-3))*u[ind_E]/rr[ind_E], color=colors[i], label="{:.2e}".format(EE[ind_E][0]) + ' GeV')
            ax.set_yscale('log')
            plt.xlabel('r (pc)')
            plt.ylabel('f_e (GeV^-1 cm^-3)')
            plt.legend()
            plt.savefig(path_name + output_type[1] + '/' + equildist_figname)
            print('MESSAGE: Equillibrium distribution plot saved at: ' + path_name + output_type[1] + '/' + equildist_figname)
        if sync or ic:
            if not (exists[2] and not overwrite[2] and exists[3] and not overwrite[3]) or temp[2] or temp[3]:
                #prepare rho-nu space0
                nu_v = np.logspace(np.log10(nu_range[0]), np.log10(nu_range[1]), mv.nnu)
                rho_v = np.linspace(rho_range[0], rho_range[1], mv.nrho)
                rhorho, nunu = np.meshgrid(rho_v, nu_v)
                
                #generate functions relevant to emission computation
                fe, f_star = pr.gen_funcs_sync_ic([rr, EE, u], f_star_params, rho_star_r)

            #prepare file name for error of synchrotron emission
            ind = output_names[2].index('synchrotron_emission.npy')
            err_sync_output_name = output_names[2][:ind] + 'err_' + output_names[2][ind:]
            #if synchrotron output does not exist or if overwrite is true
            if sync:
                if not (exists[2] and not overwrite[2]) or temp[2]:
                    #create name for sync radiation plot
                    if temp[2]:
                        sync_figname = codes[2] + '_temporary_' + output_type[2] + '.pdf'
                    else:
                        sync_figname = codes[2] +'_'+ output_type[2] + '.pdf'
                    
                    #compute synchrotron
                    def B_r_G(r):
                        return (1e-6)*B_r(r)
                    nu_sync_max = 10*(cf.e*np.max(B_r_G(rr))/(cf.c*cf.me*cf.g_per_GeV))*(np.max(EE)/cf.me)**2/(2*np.pi)
                    print('Max Frequency for Synchrotron: ', nu_sync_max, ' Hz')
                    nudSdth_syn = 0*rhorho
                    err_nudSdth_syn = 0*rhorho
                
                    for i in range(mv.nnu):
                        if nunu[i][0]<nu_sync_max:
                            for j in range(mv.nrho):
                                print('nu = ' + str(nunu[i][j]) + ' Hz')
                                print('rho = ' + str(rhorho[i][j]) + ' pc')
                                syn_flux = pr.compute_sync(nunu[i][j], rhorho[i][j], mx, Rmax, DA, funcs=[B_r_G, fe])
                                nudSdth_syn[i][j] = syn_flux[0]
                                err_nudSdth_syn[i][j] = syn_flux[1]
                            i_syncmax = i
                    np.save(output_names[2], nudSdth_syn)
                    np.save(err_sync_output_name, err_nudSdth_syn)
                    
                    print('MESSAGE: Synchrotron spectrum and morphology computed successfully.')
                    print('MESSAGE: Synchrotron spectrum and mophology saved at: ', output_names[2])
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    colors = ['m', 'r', 'orange', 'g', 'c', 'b', 'k']
                    step_nu = np.round((i_syncmax+1)/7)
                    for i in range(7):
                        ind_nu = int((i+1)*step_nu)-1
                        if ind_nu<=i_syncmax:
                            ax.plot(rhorho[ind_nu]/np.sqrt(rhorho[ind_nu]**2+DA**2), nudSdth_syn[ind_nu], color=colors[i], label="{:.2e}".format(nunu[ind_nu][0]) + ' Hz')
                    ax.set_yscale('log')
                    plt.xlabel('theta (rad)')
                    plt.ylabel('nu dS/(dnu dtheta) (erg s^-1 cm^-2 rad^-1)')
                    plt.legend()
                    plt.savefig(path_name + output_type[2] + '/' + sync_figname)
                    print('MESSAGE: Synchrotron radiation plot saved at: '+path_name + output_type[2] + '/' + sync_figname)
                else:
                    nudSdth_syn = np.load(output_names[2])
                    err_nudSdth_syn = np.load(err_sync_output_name)
                    print('MESSAGE: Synchrotron spectrum and mophology already exists and was loaded from: ', output_names[2])
            else:
                nudSdth_syn=None
            if ic:
                ind = output_names[3].index('inverse_compton_emission.npy')
                err_ic_output_name = output_names[3][:ind] + 'err_' + output_names[3][ind:]
                #if inverse compton output does not exist or if overwrite is true
                if not (exists[3] and not overwrite[3]) or temp[3]:
                    #create name for ic radiation plot
                    if temp[3]:
                        ic_figname = codes[3] + '_temporary_' + output_type[3] + '.pdf'
                    else:
                        ic_figname = codes[3] + '_' + output_type[3] + '.pdf'
                    
                    #compute inverse compton
                    nudSdth_ic = 0*rhorho
                    err_nudSdth_ic = 0*rhorho
                    E_ic_min = 1 #eV
                    nu_ic_min = E_ic_min/(cf.h_bar*2*np.pi)
                    print('MESSAGE: Minimum Frequency for Inverse Compton: ', nu_ic_min, ' Hz')

                    for i in range(mv.nnu):
                        if nunu[i][0] > nu_ic_min:
                            for j in range(mv.nrho):
                                print('nu = ' + str(nunu[i][j]) + ' Hz')
                                print('rho = ' + str(rhorho[i][j]) + ' pc')
                                ic_flux = pr.compute_ic(nunu[i][j], rhorho[i][j], mx, Rmax, DA, funcs=[rho_star_r, fe, f_star])
                                nudSdth_ic[i][j] = ic_flux[0]
                                err_nudSdth_ic[i][j] = ic_flux[1]
                        else:
                            minind_ic = i
                    minind_ic += 1
                    print('minind_ic = ', str(minind_ic))
                    np.save(output_names[3], nudSdth_ic)
                    np.save(err_ic_output_name, err_nudSdth_ic)
                    print('MESSAGE: Inverse compton spectrum and morphology computed successfully.')
                    print('MESSAGE: Inverse compton spectrum and mophology saved at: ', output_names[3])
                    step_nu = np.round((mv.nnu-minind_ic)/7)
                    print('step_nu = ',str(step_nu) )
                    fig = plt.figure()
                    ax = fig.add_subplot()
                    colors = ['m', 'r', 'orange', 'g', 'c', 'b', 'k']
                    for i in range(7):
                        ind_nu = int(i*step_nu+minind_ic)
                        print('ind_nu_'+ str(i)+ ' = '+str(ind_nu))
                        if ind_nu<=mv.nnu-1:
                            ax.plot(rhorho[ind_nu]/np.sqrt(rhorho[ind_nu]**2+DA**2), nudSdth_ic[ind_nu], color=colors[i], label="{:.2e}".format(nunu[ind_nu][0]) + ' Hz')
                    ax.set_yscale('log')
                    plt.xlabel('theta (rad)')
                    plt.ylabel('nu dS/(dnu dtheta) (erg s^-1 cm^-2 rad^-1)')
                    plt.legend()
                    plt.savefig(path_name + output_type[3] + '/' + ic_figname)
                    print('MESSAGE: Inverse Compton radiation plot saved at: '+path_name + output_type[3] + '/' + ic_figname)
                    
                else:
                    nudSdth_ic = np.load(output_names[3])
                    err_nudSdth_ic = np.load(err_ic_output_name)
                    print('MESSAGE: Inverse Compton spectrum and mophology already exists and was loaded from: ', output_names[3])
            else:
                nudSdth_ic=None
        
        else:
            nudSdth_syn=None
            nudSdth_ic=None
    
    else:
        u=None
        nudSdth_syn=None
        nudSdth_ic=None
    
    return e_spec, e_bins, gamma_spec, gamma_bins, u, nudSdth_syn, nudSdth_ic

if __name__=='__main__':
    print('name is main')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mx', help = 'Dark Matter Mass (GeV)', type = float)
    parser.add_argument('--sigma_v', help = 'Thermally Averaged Cross-section (cm^3/s)', type = float)
    parser.add_argument('--Rmax', help = 'Radius of Diffusion-zone (kpc)', type = float)
    parser.add_argument('--DA', help = 'Angular Distance (kpc)', type=float)
    parser.add_argument('--D0', help = 'Diffusion Coefficient Normalization (cm^2/s)', type = float)
    parser.add_argument('--nu_range', nargs='+', type=float,  help = '[nu_min (Hz), nu_max (Hz)]')
    parser.add_argument('--rho_range', nargs='+', type=float,  help = '[rho_min (kpc), rho_max (kpc)]')
    parser.add_argument('--B_model', nargs='+', type=str, action='append',  help = 'Model Type and Parameters for Magnetic Field')
    parser.add_argument('--rho_star_model', nargs='+', type=str, action='append', help = 'Model and Parameters for Stellar Energy Density')
    parser.add_argument('--DM_model', nargs='+', type=str, action='append', help = 'Model and Parameters for Dark Matter Density Distribution ')
    parser.add_argument('--f_star_params', nargs = '+', type=float, help= '[Average Star Temperature (K), Standard Dev. Star Temp (K)]')
    parser.add_argument('--output_names', nargs='+', type=str, help='list of names of output files')
    parser.add_argument('--file_info_names', nargs='+', type=str, help='list of names of file info files')
    parser.add_argument('--exists', nargs='+', type=str2bool, help='list of boolean existence status')
    parser.add_argument('--codes', nargs='+', type=str, help='list of 4 digit codes that label outputs' )
    parser.add_argument('--overwrite', nargs='+', type=str2bool, help='Do you want to overwrite the outputs if output already exists?')
    parser.add_argument('--temporary', nargs='+', type=str2bool, help='Do you want to save the outputs as temporary files?')
    args = parser.parse_args()
    print('arguments have been parsed')
    for i in range(len(args.B_model)):
        args.B_model[i][1:] = tuple(map(float, args.B_model[i][1:]))
        if args.B_model[i][0] == 'exp':
            #convert scale radius to pc
            args.B_model[i][2] = args.B_model[i][2]*1000
    
    for i in range(len(args.rho_star_model)):
        args.rho_star_model[i][1:] = tuple(map(float, args.rho_star_model[i][1:]))
        if args.rho_star_model[i][0] == 'exp':
            #convert scale radius to pc
            args.rho_star_model[i][2] = args.rho_star_model[i][2]*1000
    
    for i in range(len(args.DM_model)):
        args.DM_model[i][1:] = tuple(map(float, args.DM_model[i][1:]))
        if args.DM_model[i][0] == 'nfw':
            #convert scale radius to pc
            args.DM_model[i][3] = args.DM_model[i][3]*1000
    #convert f_star parameters from K to eV
    args.f_star_params = np.array(args.f_star_params) 
    args.f_star_params = cf.kb*args.f_star_params 
    args.Rmax = args.Rmax*1000
    args.DA = args.DA*1000
    args.rho_range = np.array(args.rho_range)*1000
    print('call main funciton')
    #sync_nudSdth
    e_spec, e_bins, gamma_spec, gamma_bins, u, sync_nudSdth, ic_nudSdth = \
        main(args.mx, args.sigma_v, args.Rmax, args.DA, args.D0, args.nu_range, args.rho_range, \
             args.B_model, args.rho_star_model, args.DM_model, args.f_star_params, args.output_names, \
                 args.file_info_names, args.exists, args.codes, overwrite=args.overwrite, temp=args.temporary)

