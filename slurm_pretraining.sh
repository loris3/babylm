#!/bin/bash
# script name: pretrain.sh
  

 
source /etc/profile.d/modules.sh

# Create a permanent environment with the name "my_new_permanent_environment"
export ENV_MODE="permanent"
export ENV_NAME="pretraining_permanent_environment"
# This environment is now stored in $HOME/venvs/my_new_permanent_environment
 
# Load module miniforge3
module load miniforge
 
cd /data/loriss21dm/babylm
# E.g. to install networkx
conda install -y environment.yml
 
# Run your python script
python ./main.py
 
# Cleanup
module purge

