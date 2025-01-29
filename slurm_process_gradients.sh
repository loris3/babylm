#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="influence computation"
#SBATCH --comment="RoBERTa Training Data Influence Experiments. Runs after gradient extraction for checkpoint $3"
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=795GB
#SBATCH --cpus-per-task=36
#SBATCH --nodelist=dgx-h100-em2

source /etc/profile.d/modules.sh

# Create a permanent environment with the name "my_new_permanent_environment"
export ENV_MODE="permanent"
export ENV_NAME="babylm_venv"
# This environment is now stored in $HOME/venvs/my_new_permanent_environment
 
# Load module miniforge3
module load miniforge

# df -h $TMPDIR/gradients
# rm -rf $TMPDIR/gradients


# df -h $TMPDIR/gradients

conda env update --file environment.yml

# Run your python script
python process_gradients.py $1 $2 $3 --mode=mean
 
# Cleanup
module purge

