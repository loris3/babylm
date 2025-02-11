#!/bin/bash
# script name: pretrain.sh
#SBATCH --job-name="lm pretraining experiments"
#SBATCH --comment="pretraining llama and bert models for training data influence experiments"
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --ntasks=1
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
python pretrain.py $1 $2 --model_type=$3 
 
# Cleanup
module purge

