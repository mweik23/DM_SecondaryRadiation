#/usr/bin/env python
#

import sys, os, time, fileinput, re
import numpy as np
import glob
import Secondary_radiation.scripts.meta_variables as mv
import Secondary_radiation.scripts.manipulate_text as mt

print('make job is running')

#parameters
mx_min = 6
mx_max = 500
mx_num = 20
#mx_list = np.logspace(np.log10(mx_min), np.log10(mx_max), mx_num)
mx_list = [6.0]
sigma_v = 2.2e-26
Rmax = 50
DA = 780 
D0_list = [3e27]
nu_range = 1e5, 1e24
rho_range = .1, 35
B_model = [['exp', 10, 1.5], ['exp', 8, 20]] 
rho_star_model = [['exp', 8, 4.3]]
DM_model = [['nfw', 0.43, 1.25, 16.5]]
f_star_params = 5500, 1000

#options
equil = True
sync = True
ic = True
overwrite = 5*[False]
temp = 5*[False]

#get path
this_path_full = os.path.realpath(__file__)
m = re.search('/makejob.*\.py$', this_path_full)
this_file = m.group(0)
working_dir_full = this_path_full.split(this_file)[0]
username = working_dir_full.split('/')[-2]
path_to_username = working_dir_full.split(username)[0]+username

output_type = ['electron_spectrum', 'equillibrium_distribution', 'synchrotron_emission', \
               'inverse_compton_emission', 'photon_spectrum']
# =============================================================================
# username="mjw283"
# working_dir = "eventgen_wd"
# working_dir_full = "/het/p4/" + username+ "/" +working_dir    
# =============================================================================

#make folder for python scripts. Should I put the absolute path here?
executedir = "scan_exec"
os.system("rm -rf "+executedir)
os.system("mkdir "+executedir)

#open file to contain submission commands for jdl files
dofile = open(executedir+"/do_all.src",'w')

#RESULTS_DIR = working_dir_full + "/outputdir"
#make directory if it doesn't already exist
#os.system("mkdir -p "+RESULTS_DIR)
cl_options_base = " --sigma_v " + str(sigma_v) + " --Rmax " + str(Rmax) + " --DA " + str(DA) \
    +" --nu_range " + str(nu_range[0]) + ' ' + str(nu_range[-1]) + ' --rho_range ' \
        + str(rho_range[0]) + ' ' + str(rho_range[-1])
for model in B_model:
    cl_options_base += " --B_model"
    for component in model:
        cl_options_base += ' ' + str(component) 
for model in rho_star_model:
    cl_options_base += " --rho_star_model"
    for component in model:
        cl_options_base += ' ' + str(component) 
for model in DM_model:
    cl_options_base += " --DM_model"
    for component in model:
        cl_options_base += ' ' + str(component)

cl_options_base += ' --f_star_params ' + str(f_star_params[0]) + ' ' + str(f_star_params[1])

for mx in mx_list:
    mx = np.round(mx, 1)
    cl_options_mx = " --mx " + str(mx) + cl_options_base
    for D0 in D0_list:
        cl_options = "--D0 " + str(D0) + cl_options_mx
        print('mx = ' + str(mx) + ' D0 = ' + str(D0))
        
        #manage filenames
        file_info = []
    
        for i in range(len(output_type)):
            file_info.append(mt.gen_fileinfo(i, mx=mx, channel='bb_bar', sigma_v=sigma_v, Rmax=Rmax, DA=DA, D0=D0, nu_range=nu_range, \
                                             rho_range=rho_range, B_model=B_model, rho_star_model=rho_star_model, DM_model=DM_model, \
                                                f_star_params=f_star_params, nr=mv.nr, nE = mv.nE, nrho=mv.nrho, nnu=mv.nnu,))
        output_names = [] 
        file_info_names =  []
        exists = []
        codes = []
        #check if all types of output are turned on. For any outputs that are not turned on, 
        #do not check_txt_files and put None in the place of the outputs of check_txt_files 
        if not equil:
            file_info[1] = None
        if not sync:
            file_info[2] = None
        if not ic:
            file_info[3] = None
        for i in range(len(file_info)):
            if file_info[i] == None:
                output_name = None
                file_info_name = None
                exists_bool = None
                code = None
            else:
                output_name, file_info_name, exists_bool, code = mt.check_txt_files(file_info[i], working_dir_full + '/Secondary_radiation/', temp=temp[i])
            #append these to the lists initiated on lines 91 to 94
            output_names.append(output_name)
            file_info_names.append(file_info_name)
            exists.append(exists_bool)
            codes.append(code)
        cl_options += ' --output_names'
        for output_name in output_names:
            cl_options += ' ' + output_name
        cl_options += ' --file_info_names'
        for file_info_name in file_info_names:
            cl_options += ' ' + file_info_name
        cl_options += ' --exists'
        for exists_bool in exists:
            cl_options += ' ' + str(exists_bool)
        cl_options += ' --codes'
        for code in codes:
            cl_options += ' ' + code
        cl_options+= ' --overwrite'
        for b in overwrite:
            cl_options+= ' ' + str(b) 
        cl_options += ' --temporary'
        for b in temp:
            cl_options += ' ' + str(b)
        
        runname = 'compute_signal_' + 'mx_' + str(mx) +'_D0_' + str(D0)
        
        #define the local scratch folder
        localdir = "$_CONDOR_SCRATCH_DIR"
        #write out script for job execution
        execfilename = "exec_%04i"%mx + '_' + str(D0) + '.sh'
        executefile = open(executedir+"/"+execfilename,'w')
        executefile.write("#!/bin/bash\n")
        #executefile.write("export VO_CMS_SW_DIR=\"/cms/base/cmssoft\"\n")
        #executefile.write("export COIN_FULL_INDIRECT_RENDERING=1\n")
        #executefile.write("export SCRAM_ARCH=\"slc5_amd64_gcc434\"\n")
        #executefile.write("source $VO_CMS_SW_DIR/cmsset_default.sh\n")
    
        #move to local directory on the node
        #executefile.write("cd "+localdir+"\n")
        
        executefile.write('python ' + working_dir_full+ '/Secondary_radiation/scripts/compute_signal.py ' + cl_options + "\n")
    
        executefile.close()
    
        os.system("chmod u+x "+executedir+"/"+execfilename)
    
        #write out jdl script for job submission
        jdlfilename = "exec_%04i"%mx + '_' + str(D0) + '.jdl.base'
        jdlfile = open(executedir+"/"+jdlfilename,'w')
        jdlfile.write("universe = vanilla\n")
        jdlfile.write("+AccountingGroup = \"group_rutgers."+username+"\"\n")
        jdlfile.write("Executable = " + working_dir_full + "/"+executedir+"/"+execfilename+"\n")
        jdlfile.write("getenv = True\n")
        jdlfile.write("should_transfer_files = NO\n")
        jdlfile.write("Arguments = \n")
        jdlfile.write("Output = " + path_to_username +"/condor/"+runname+".out\n")
        jdlfile.write("Error = "+ path_to_username+ "/condor/"+runname+".err\n")
        jdlfile.write("Log = "+ path_to_username + "/condor/script.condor\n")
        jdlfile.write("Queue 1\n")
        jdlfile.close()
    
        dofile.write("condor_submit "+jdlfilename+"\n")
dofile.close()
