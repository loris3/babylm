#!/bin/bash
# script name: pretrain.sh
#SBATCH --job-name="pretraining handcrafted curriculum"
#SBATCH --comment="LLama Training Data Influence Experiments."
#SBATCH --time=0-12:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --nodelist=dgx-h100-em2
source /etc/profile.d/modules.sh

# Create a permanent environment with the name "my_new_permanent_environment"
export ENV_MODE="permanent"
export ENV_NAME="babylm_venv"
# This environment is now stored in $HOME/venvs/my_new_permanent_environment

# Load module miniforge3
module load miniforge
conda env update --file environment.yml




# E.g. to install networkx



# Run your python script
python pretrain.py loris3/stratified_equitoken_10m_curriculum random.pt --model_type="llama"
 
# Cleanup
module purge

