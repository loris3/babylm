#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="influence computation"
#SBATCH --comment="Olmo2 Training Data Influence Experiments. Runs after gradient extraction for checkpoint $3"
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1750GB
#SBATCH --cpus-per-task=32
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

# conda env update --file environment.yml

# Run your python script
python process_gradients.py $1 $2 $3 --mode=mean --dataset_test_split=train --dataset_train_split=train
 
# Cleanup
module purge

