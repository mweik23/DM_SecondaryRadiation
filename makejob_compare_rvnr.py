#/usr/bin/env python
#

import sys, os, time, fileinput, re
import numpy as np
import glob
import Secondary_radiation.scripts.meta_variables as mv
import Secondary_radiation.scripts.manipulate_text as mt

print('make job is running')

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

#create runname
runname = 'compare_rvnr'
        
#define the local scratch folder
localdir = "$_CONDOR_SCRATCH_DIR"
#write out script for job execution
execfilename = runname + '.sh'
executefile = open(executedir+"/"+execfilename,'w')
executefile.write("#!/bin/bash\n")
#executefile.write("export VO_CMS_SW_DIR=\"/cms/base/cmssoft\"\n")
#executefile.write("export COIN_FULL_INDIRECT_RENDERING=1\n")
#executefile.write("export SCRAM_ARCH=\"slc5_amd64_gcc434\"\n")
#executefile.write("source $VO_CMS_SW_DIR/cmsset_default.sh\n")
    
#move to local directory on the node
#executefile.write("cd "+localdir+"\n")
        

executefile.write('python ' + working_dir_full+ '/Secondary_radiation/scripts/'+runname +'.py'+ '\n')
    
executefile.close()
    
os.system("chmod u+x "+executedir+"/"+execfilename)
    
#write out jdl script for job submission
jdlfilename = runname + '.jdl.base'
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
    
print('job  created')
dofile.close()
