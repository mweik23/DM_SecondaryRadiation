#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 13:17:33 2021

@author: mitchellweikert
"""

import numpy as np
import hepmcio_modified as hepmcio
import os
import matplotlib.pyplot as plt

me = 5.10999*10**(-4)  
this_path = os.path.realpath(__file__)
in_path = this_path.split('Secondary_radiation/')[0]
out_path = this_path.split('scripts')[0]

Events_path = in_path + 'outputdir/'

def get_hepmc_names(mx, Events_path=Events_path):
    #gather the run directories into a list
    run_list = os.listdir(Events_path)
    run_set = []
    for run in run_list:
        #if the run directory is a tar ball then unzip and give it the name that it had minus the .tar.gz
        if run.split('.')[-1] == 'gz' and run.split('.')[-2]=='tar':
            print('unzipping ' + run)
            run_dir = run.split('.tar.gz')[0]
            os.system('mkdir ' + Events_path + run_dir)
            os.system('tar -xvf ' + Events_path + run + ' -C ' + Events_path+run_dir + ' --strip-components=1')
            os.system('rm ' + Events_path + run)
        #if it is just a gz then gunzip it. Not sure if this will give the directory the correct name after unzipping
        elif run.split('.')[-1] == 'gz':
            os.system('gunzip '+ Events_path + run)
   # Now that run directories have been unzipped, regather the names of the contents in outputdir
    run_list = os.listdir(Events_path)
    for run in run_list: 
        # if the item is a directory then 
        if os.path.isdir(Events_path + run):
            file_list = os.listdir(Events_path + run)  # gather the items in the run directory
            # search through the items in the run directory for the hepmc file(s)
            for file_name in file_list:
                dsp_file = file_name.split('.')
                if 'hepmc' in dsp_file:
                    tag_name = file_name.split('_')[0]
                    str_mx = str(mx)
                    #if hepmc file starts with 'E<mx>' then add the run directory to run_set and stop searching that directory
                    if tag_name == 'E' + str_mx:
                        run_set.append(run)
                    break

    hepmc_path_set = []
    hepmcgz_path_set = []
    hepmc_full_path_set = []
    
    if run_set == []:
        print('ERROR: No hepmc file for requested energy')
    else:
        #for each run of the requested mass, search the files within
        for run in run_set:
            file_list = os.listdir(Events_path + run)
            for file_name in file_list:
                dsp_file = file_name.split('.')
                #add the unzipped hepmc file path to hepmc_path_set and then stop searching that directory. 
                #Note: path starts from run directory. For example 'run_500.0/E500.0_pythia8_events.hepmc'
                if dsp_file[-1] == 'hepmc':
                    hepmc_path_set.append(run + '/' + file_name)
                    #determine the zipped version of this hepmc path and look for it in hepmcgz_path_set
                    unz2z = run + '/' + file_name + '.gz'
                    # if the zipped file has already been stored in hepmcgz_path_set then remove it
                    if unz2z in hepmcgz_path_set:
                        hepmcgz_path_set.remove(unz2z)
                    break
                #if it comes accross a zipped hepmc file then add it to hepmcgz_path_set
                elif dsp_file[-2] == 'hepmc' and dsp_file[-1] == 'gz':
                    hepmcgz_path_set.append(run + '/' + file_name)
    
    #unzip the zipped hepmc files and add the unzipped path to hepmc_path_set
    for zipped_path in hepmcgz_path_set:
        os.system('gunzip '+ Events_path + zipped_path)
        unzipped_path = zipped_path.replace('.gz', '')
        hepmc_path_set.append(unzipped_path)
    
    #for each unzipped relative, path construct the absolute path and add it to hepmc_full_path_set
    for unzipped_path in hepmc_path_set:
        hepmc_full_path_set.append(Events_path + unzipped_path)
    
    return hepmc_full_path_set

def hepmc_to_list(file_path):
    reader = hepmcio.HepMCReader(file_path)
    evtnum = 0
    enum = 0
    gammanum = 0
    eE = []
    gammaE = []
    while True:
        evtnum += 1
        evt = reader.next()
        if not evt:
            evtnum -= 1
            break
        if evtnum%1000==0:
            print('event number: ', evtnum)
            print('number of particles: ', len(evt.particles))
        for i in range(len(evt.particles)):
            if evt.particles[i+1].status == 1:
                if np.abs(evt.particles[i+1].pid)==11:
                    enum+=1
                    eE.append(evt.particles[i+1].mom[3])
                if np.abs(evt.particles[i+1].pid)==22:
                    gammanum+=1
                    gammaE.append(evt.particles[i+1].mom[3])
    return eE, gammaE, evtnum

def get_spectra(eE, gammaE, numev, mx, plot_e=False, plot_gamma=False):
    eE_ar = np.array(eE)
    gammaE_ar = np.array(gammaE)
    max_eE = np.max(eE_ar)
    if max_eE>=mx:
        print('electron energies are larger than or equal to dark matter mass. Check Pythia output.')
    e_dsy, e_bins = np.histogram(eE_ar, np.logspace(np.log10(me), np.log10(mx), 300), density=True)
    e_spec = e_dsy*len(eE_ar)/numev

    if plot_e:   
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(e_bins[1:], e_spec)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlabel('E (GeV)')
        plt.ylabel('dN_e/dE (GeV^-1)')
        plt.title('m_x = '+ str(mx) + ' GeV')
        plt.savefig(out_path + 'figs/' + 'mx' + str(mx) + '_plot_espec.pdf')

    gamma_dsy, gamma_bins = np.histogram(gammaE_ar, np.logspace(np.log10(np.min(gammaE_ar)), np.log10(np.max(gammaE_ar)), 300), density=True)
    gamma_spec = gamma_dsy*len(gammaE_ar)/numev

    if plot_gamma:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(gamma_bins[:-1], gamma_spec)
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xlabel('E (GeV)')
        plt.ylabel('dN_gamma/dE (GeV^-1)')
        plt.title('m_x = '+ str(mx) + ' GeV')
        plt.savefig(out_path + 'figs/' + 'mx' + str(mx) + '_plot_gammaspec.pdf')
    
    return e_bins, e_spec, gamma_bins, gamma_spec

if __name__ == '__main__':
    hepmc_paths = get_hepmc_names(500.0)