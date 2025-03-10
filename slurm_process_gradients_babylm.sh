#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="influence computation"
#SBATCH --comment="Training Data Influence Experiments. Loads gradients from network share DIRECTLY TO RAM"
#SBATCH --time=0-06:00:00
#SBATCH --ntasks=1
#SBATCH --mem=650GB
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=galadriel,dgx-h100-em2
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
python process_gradients.py $1 $2 $6 --dataset_train_split=$3 --dataset_test=$4 --dataset_test_split=$5 --mode=mean  --batch_size=1000 --gradients_per_file=1000
 
# Cleanup
module purge

