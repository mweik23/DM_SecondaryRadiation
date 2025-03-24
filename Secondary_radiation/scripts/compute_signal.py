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
               'photon_spectrum']
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
        
def main(mx, sigma_v, rmax, D0, nu_range, nu_space, thx_range, thy_range, DM_model, astro_model, spherical_ave, output_names,\
         file_info_names, exists, codes, overwrite=[False for ouput in output_type], \
             temp=[False for output in output_type], channel='bb_bar', E_spacing='log'):
    print('main function has started running')
    equil = True
    sync = True
    
    iesp = output_type.index('electron_spectrum')
    ipho = output_type.index('photon_spectrum')
    iequ = output_type.index('equillibrium_distribution')
    isyn = output_type.index('synchrotron_emission')
    if output_names[iequ] is None:
        equil=False
    if output_names[isyn] is None:
        sync = False
    #construct espec bins name
    espec_name_split = output_names[iesp].split('/')
    
    if temp[iesp]:
        espec_name_split[-1] = codes[iesp] + '_temporary_' + espec_name_split[-2] + '_bins.npy'
        espec_fig_name = codes[iesp] + '_temporary_' + output_type[iesp] + '.pdf'
    else:
        espec_name_split[-1] = codes[iesp] + '_' + espec_name_split[-2] + '_bins.npy'
        espec_fig_name = codes[iesp] + '_' + output_type[iesp] + '.pdf'
    espec_bins_name = ''
    for part in espec_name_split:
        if not (part ==''):
            espec_bins_name += '/' + part
    #constuct path for bins array
    gammaspec_name_split = output_names[ipho].split('/')
    code = gammaspec_name_split[-1].split('_')[0]
    if temp[ipho]:
        gammaspec_name_split[-1] = codes[ipho] + '_temporary_' + gammaspec_name_split[-2] + '_bins.npy'
        gammaspec_fig_name = codes[ipho] + '_temporary_' + output_type[ipho] + '.pdf'
    else:
        gammaspec_name_split[-1] = codes[ipho] + '_' + gammaspec_name_split[-2] + '_bins.npy'
        gammaspec_fig_name = codes[ipho] + '_' + output_type[ipho] + '.pdf'
    gammaspec_bins_name = ''
    for part in gammaspec_name_split:
        if not (part==''):
            gammaspec_bins_name += '/' + part
    
    # If both electron spectrum and photon spectrum exist and I do not want to overwrite them, 
    #then I don't have to process the hepmc files and can just load the electron spectrum and photon spectrum
    if exists[iesp] and exists[ipho] and not overwrite[iesp] and not overwrite[ipho] and not temp[iesp] and not temp[ipho]:
        print('MESSAGE: Electron spectrum and photon spectrum already exists for these parameters.' + \
              ' Loading electron spectrum from: ' + output_names[iesp] + '. Loading photon spectrum from ' + output_names[ipho])
        e_spec = np.load(output_names[iesp])
        e_bins = np.load(espec_bins_name)
        gamma_spec = np.load(output_names[ipho])
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
            if not (exists[ipho] and not overwrite[ipho]) or temp[ipho]:   
                print('MESSAGE: Photon spectrum does not exist. Saving photon spectrum and bins')
                
                #save gamma spectrum and bins
                np.save(output_names[ipho], gamma_spec)
                np.save(gammaspec_bins_name, gamma_bins)
                print('MESSAGE: Photon spectrum saved at: ', output_names[ipho])
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
                plt.savefig(path_name + output_type[ipho] + '/' + gammaspec_fig_name)
                
                
                print('MESSAGE: Photon spectrum plot saved at: '+ path_name + output_type[ipho] + '/' + gammaspec_fig_name)
            else:
                print('MESSAGE: Photon spectrum exists. Loading spectrum and bins from ' + output_names[ipho] + ' and ' + gammaspec_bins_name + ', respectively.')
                gamma_spec = np.load(output_names[ipho])
                gamma_bins = np.load(gammaspec_bins_name)
            #if electron spectrum does not exist
            if not (exists[iesp] and not overwrite[iesp]) or temp[iesp]:
                print('MESSAGE: Electron spectrum does not exist. Saving electton spectrum and bins')
                #save e spectrum and bins
                np.save(output_names[iesp], e_spec)
                np.save(espec_bins_name, e_bins)
                print('MESSAGE: Electron spectrum saved at: ', output_names[iesp])
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
                plt.savefig(path_name + output_type[iesp]+ '/' + espec_fig_name)
                
                print('MESSAGE: Electron spectrum plot saved at: '+ path_name + output_type[iesp] + '/' + espec_fig_name)
            else:
                print('MESSAGE: Electron spectrum exists. Loading spectrum and bins from ' + output_names[iesp] + ' and ' + espec_bins_name + ', respectively.')
                e_spec = np.load(output_names[iesp])
                e_bins = np.load(espec_bins_name)
    
    if equil:
        #if equillibrium distribution result already exist and I do not want to overwrite it, then load  equillibrium dist
        if exists[iequ] and not overwrite[iequ] and not temp[iequ]:
            print('MESSAGE: Equillibrium distribution for these variables exists at: ', output_names[iequ])
            u = np.load(output_names[iequ])
            #check if shape is correct
            if u.shape == (mv.nE, mv.nr):
                print('MESSAGE: Equillibrium phase-space distribution successfully loaded')
                rr, EE, dr, dE = mv.grid(mv.nr+3, mv.nE+2, [0, rmax], [cf.me, mx], E_spacing=E_spacing)
                rr = rr[1:-1, 1:-2]
                EE = EE[1:-1, 1:-2]
            else:
                statement = 'MESSAGE: The shape of the loaded equillibrium phase-space distribution does not match grid shape from meta variables.'
                statement = statement + ' Change the grid-size in meta_variables.py so that it matches the loaded file, or overwrite the existing electron equillibrium distribution.'
                print(statement)
        #otherwise compute equillibrium distribution
        else:
            print(mx, sigma_v, D0, rmax, DM_model, E_spacing)
            rr, EE, u = pr.find_equillibrium(mx, sigma_v, D0, rmax, e_spec, e_bins, am, spherical_ave=spherical_ave, DM_model=DM_model, E_spacing=E_spacing)
            np.save(output_names[iequ], u)
            print('MESSAGE: Equillibrium distribution computed and saved at: ' + output_names[iequ])
            
            if temp[iequ]:
                equildist_figname = codes[iequ] + '_temporary_' + output_type[iequ] + '.pdf'
            else:
                equildist_figname = codes[iequ] + '_' + output_type[iequ] + '.pdf'
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
            print('MESSAGE: Equillibrium distribution plot saved at: ' + path_name + output_type[iequ] + '/' + equildist_figname)
                
        #if synchrotron output does not exist or if overwrite is true
        if sync:
            if not (exists[isyn] and not overwrite[isyn]) or temp[isyn]:
                fe = pr.smooth_fe([rr, EE, u])
                #prepare rho-nu space0
                if nu_space == 'lin':
                    nu_v = np.linspace(nu_range[0], nu_range[1], mv.nnu)
                elif nu_space == 'log':
                    nu_v = np.logspace(np.log10(nu_range[0]), np.log10(nu_range[1]), mv.nnu)
                else:
                    print('nu_space not recognized')
            
                #prepare file name for error of synchrotron emission
                ind = output_names[isyn].index('synchrotron_emission.npy')
                err_sync_output_name = output_names[isyn][:ind] + 'err_' + output_names[isyn][ind:]
                #create name for sync radiation plot
                if temp[isyn]:
                    sync_figname = codes[isyn] + '_temporary_' + output_type[isyn] + '.pdf'
                else:
                    sync_figname = codes[isyn] +'_'+ output_type[isyn] + '.pdf'
                
                xv = np.linspace(-thx_range[0]*am.DA, thx_range[0]*am.DA, 2*mv.nx)
                yv = np.linspace(-thy_range[0]*am.DA, thy_range[0]*am.DA, 2*mv.ny)
                xv = xv[mv.nx:]
                yv = yv[mv.ny:]
                xx, yy = np.meshgrid(xv,yv)
                rhorho = np.sqrt(xx**2+yy**2)
                phiphi = np.arctan(yy/xx)
                
                res = np.array([[[pr.compute_sync(nu, rhorho[j,i], phiphi[j,i], mx, rmax, am.DA, am.beta, B=am.B, fe_func=fe) \
                                 for i in range(mv.nx)] for j in range(mv.ny)] for nu in nu_v])
                nudSdth_syn = np.einsum('jkli', res)
                np.save(output_names[isyn], nudSdth_syn[0])
                np.save(err_sync_output_name, nudSdth_syn[1])
                print('MESSAGE: Synchrotron spectrum and morphology computed successfully.')
                print('MESSAGE: Synchrotron spectrum and mophology saved at: ', output_names[isyn])
# =============================================================================
#                 fig = plt.figure()
#                 ax = fig.add_subplot()
#                 colors = ['m', 'r', 'orange', 'g', 'c', 'b', 'k']
#                 step_phi = np.round((mv.nphi+1)/7)
#                 ind_nu = int(np.round(mv.nnu/2))
#                 for i in range(7):
#                     ind_phi = step_phi*i
#                     ax.plot(rho_v/np.sqrt(rho_v**2+DA**2), nudSdth_syn[ind_nu, :,  ind_phi], \
#                             color=colors[i], label=str(np.round(phi_v[ind_phi], 2)) + ' rad')
#                 ax.set_yscale('log')
#                 plt.xlabel('theta (rad)')
#                 plt.ylabel('nu dS/(dnu dtheta) (erg s^-1 cm^-2 rad^-1)')
#                 plt.legend()
#                 plt.savefig(path_name + output_type[isyn] + '/' + sync_figname)
#                 print('MESSAGE: Synchrotron radiation plot saved at: '+path_name \
#                       + output_type[2] + '/' + sync_figname)
# =============================================================================
            else:
                nudSdth_syn = np.load(output_names[isyn])
                err_nudSdth_syn = np.load(err_sync_output_name)
                print('MESSAGE: Synchrotron spectrum and mophology already exists and was loaded from: ', output_names[isyn])
        else:
            nudSdth_syn=None
    
    else:
        u=None
        nudSdth_syn=None
    
    return e_spec, e_bins, gamma_spec, gamma_bins, u, nudSdth_syn

if __name__=='__main__':
    print('name is main')
    parser = argparse.ArgumentParser()
    parser.add_argument('--mx', help = 'Dark Matter Mass (GeV)', type = float)
    parser.add_argument('--sigma_v', help = 'Thermally Averaged Cross-section (cm^3/s)', type = float)
    parser.add_argument('--rmax', help = 'Radius of Diffusion-zone (kpc)', type = float)
    parser.add_argument('--D0', help = 'Diffusion Coefficient Normalization (cm^2/s)', type = float)
    parser.add_argument('--nu_range', nargs='+', type=float,  help = '[nu_min (Hz), nu_max (Hz)]')
    parser.add_argument('--thx_range', nargs='+', type=float,  help = '(rad)')
    parser.add_argument('--thy_range', nargs='+', type=float,  help = '(rad)')
    parser.add_argument('--nu_space', type=str, help='options are log or lin')
    parser.add_argument('--DM_model', nargs='+', type=str, action='append', help = 'Model and Parameters for Dark Matter Density Distribution ')
    parser.add_argument('--astro_model', type=str, help='name of astro model package')
    parser.add_argument('--output_names', nargs='+', type=str, help='list of names of output files')
    parser.add_argument('--file_info_names', nargs='+', type=str, help='list of names of file info files')
    parser.add_argument('--exists', nargs='+', type=str2bool, help='list of boolean existence status')
    parser.add_argument('--spherical_ave', type=str, help='method of taking spherical average')
    parser.add_argument('--codes', nargs='+', type=str, help='list of 4 digit codes that label outputs' )
    parser.add_argument('--overwrite', nargs='+', type=str2bool, help='Do you want to overwrite the outputs if output already exists?')
    parser.add_argument('--temporary', nargs='+', type=str2bool, help='Do you want to save the outputs as temporary files?')
    args = parser.parse_args()
    print('arguments have been parsed')
    
    for i in range(len(args.DM_model)):
        args.DM_model[i][1:] = tuple(map(float, args.DM_model[i][1:]))
        if args.DM_model[i][0] == 'nfw':
            #convert scale radius to pc
            args.DM_model[i][3] = args.DM_model[i][3]*1000
    #convert f_star parameters from K to eV
    args.rmax = args.rmax*1000
    am = __import__(args.astro_model)
    
    print('call main funciton')
    #sync_nudSdth
    e_spec, e_bins, gamma_spec, gamma_bins, u, sync_nudSdth = \
        main(args.mx, args.sigma_v, args.rmax, args.D0, args.nu_range, args.nu_space,\
             args.thx_range, args.thy_range, args.DM_model, args.astro_model, args.spherical_ave, args.output_names, args.file_info_names,\
                 args.exists, args.codes, overwrite=args.overwrite, temp=args.temporary)

