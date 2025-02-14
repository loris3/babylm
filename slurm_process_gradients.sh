#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="influence computation"
#SBATCH --comment="Training Data Influence Experiments. Runs after gradient extraction for checkpoint $3"
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=500GB
#SBATCH --cpus-per-task=24
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
python process_gradients.py $1 $2 $6 --dataset_train_split=$3 --dataset_test=$4 --dataset_test_split=$5 --mode=mean  --batch_size=10
 
# Cleanup
module purge

