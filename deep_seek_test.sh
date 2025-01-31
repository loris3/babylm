#!/bin/bash
# script name: deep_seek_test.sh
#SBATCH --job-name="gradiend extraction"
#SBATCH --comment="test of deepseek-ai/DeepSeek-V3 inference"
#SBATCH --time=0-01:00:00

#SBATCH --ntasks=1
#SBATCH --mem=1000G
#SBATCH --cpus-per-task=100
#SBATCH --nodelist=dgx-h100-em2
source /etc/profile.d/modules.sh

# Create a permanent environment with the name "my_new_permanent_environment"
export ENV_MODE="permanent"
export ENV_NAME="babylm_venv"
# This environment is now stored in $HOME/venvs/my_new_permanent_environment
 
# Load module miniforge3
module load miniforge

# cp -r /srv/home/users/loriss21cs/babylm/gradients $TMPDIR/gradients
# df -h $TMPDIR
# ls $TMPDIR/gradients



# conda env update --file environment.yml

# Run your python script
python deep_seek_test.py
 
# Cleanup
module purge

