#!/bin/bash
# script name: pretrain.sh
#SBATCH --job-name="pretraining dirac_positive_only curriculum"
#SBATCH --comment="RoBERTa Training Data Influence Experiments."
#SBATCH --time=0-5:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=40
#SBATCH --nodelist=dgx-h100-em2
source /etc/profile.d/modules.sh

# Create a permanent environment with the name "my_new_permanent_environment"
export ENV_MODE="permanent"
export ENV_NAME="bayblm"
# This environment is now stored in $HOME/venvs/my_new_permanent_environment
 
# Load module miniforge3
module load miniforge
 
# E.g. to install networkx
# conda env update --file environment.yml
conda activate 

# Run your python script
python pretrain.py loris3/stratified_10m_curriculum dirac_positive_only.pt
 
# Cleanup
module purge

