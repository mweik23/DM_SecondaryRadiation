#/usr/bin/env python
#

import sys, os, time, fileinput
import numpy as np
import glob

print('make job is running')
#username
username="mjw283"
working_dir = "eventgen_wd"
    
#make folder for python scripts
executedir = "scan_exec"
os.system("rm -rf "+executedir)
os.system("mkdir "+executedir)

#open file to contain submission commands for jdl files
dofile = open(executedir+"/do_all.src",'w')

#datasets = sys.argv[1] # gd1, gaia1, gaia3 or jhelum
madgraphfolder = 'MG5_aMC_v3_1_0'
madgraph_location='/het/p4/' + username + '/' +working_dir+ '/'+madgraphfolder+'.tar.gz'

RESULTS_DIR = "/het/p4/" + username + "/" + working_dir + "/outputdir"
#make directory if it doesn't already exist
os.system("mkdir -p "+RESULTS_DIR)

energy_list = np.linspace(5,50,2)

for energy in energy_list:
    energy = int(energy)
    print(energy)
    runname = 'eebb_'+str(energy)
    
    #define the local scratch folder
    localdir = "$_CONDOR_SCRATCH_DIR"
    #write out script for job execution
    execfilename = "exec_%04i.sh"%energy
    executefile = open(executedir+"/"+execfilename,'w')
    executefile.write("#!/bin/bash\n")
    executefile.write("export VO_CMS_SW_DIR=\"/cms/base/cmssoft\"\n")
    executefile.write("export COIN_FULL_INDIRECT_RENDERING=1\n")
    executefile.write("export SCRAM_ARCH=\"slc5_amd64_gcc434\"\n")
    executefile.write("source $VO_CMS_SW_DIR/cmsset_default.sh\n")

    #move to local directory on the node
    executefile.write("cd "+localdir+"\n")
    #bring tarball over to the node
    executefile.write("cp "+madgraph_location+" ./"+"\n")
    #unpack taball
    executefile.write("tar -xzvf "+madgraphfolder+".tar.gz \n")
    #cd to the local version of madgraph's eeBB directory
    executefile.write("cd "+madgraphfolder+"/eebb \n")
    #update the run file to have the correct energy for the e-e+ beams
    executefile.write("sed -i 's/ENERGYVALUE/"+str(energy)+"/g' Cards/run_card.dat \n")
    #run the local version of MadGraph's eeBB
    executefile.write("./bin/generate_events \n")
    #copy the output file back home to your resultsdirectory
    executefile.write("cp Events/run_01/*_pythia8_events.hepmc.gz "+RESULTS_DIR+"\n")
    executefile.close()

    os.system("chmod u+x "+executedir+"/"+execfilename)

    #write out jdl script for job submission
    jdlfilename = "exec_%04i.jdl.base"%energy
    jdlfile = open(executedir+"/"+jdlfilename,'w')
    jdlfile.write("universe = vanilla\n")
    jdlfile.write("+AccountingGroup = \"group_rutgers."+username+"\"\n")
    jdlfile.write("Executable = /het/p4/"+username+"/" + working_dir + "/"+executedir+"/"+execfilename+"\n")
    jdlfile.write("getenv = True\n")
    jdlfile.write("should_transfer_files = NO\n")
    jdlfile.write("Arguments = \n")
    jdlfile.write("Output = /het/p4/"+username+"/condor/"+runname+".out\n")
    jdlfile.write("Error = /het/p4/"+username+"/condor/"+runname+".err\n")
    jdlfile.write("Log = /het/p4/"+username+"/condor/script.condor\n")
    jdlfile.write("Queue 1\n")
    jdlfile.close()

    dofile.write("condor_submit "+jdlfilename+"\n")
dofile.close()
