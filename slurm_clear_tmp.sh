#!/bin/bash
# script name: delete_tmp_dir.sh
#SBATCH --job-name="tmp dir test"

#SBATCH --time=0-00:05:00
#SBATCH --ntasks=1
#SBATCH --mem=1GB
#SBATCH --cpus-per-task=2
#SBATCH --nodelist=dgx-h100-em2

source /etc/profile.d/modules.sh

# Create a permanent environment with the name "my_new_permanent_environment"
export ENV_MODE="permanent"
export ENV_NAME="bayblm"
# This environment is now stored in $HOME/venvs/my_new_permanent_environment
 
# Load module miniforge3
module load miniforge

ls -lha $TMPDIR
# rm -rf $TMPDIR/gradients
df -h $TMPDIR

# # df -h $TMPDIR/gradients

# conda env update --file environment.yml
# conda activate 
# # Run your python script
# df -h $TMPDIR/gradients
# rm -rf $TMPDIR/gradients
# df -h $TMPDIR/gradients

 
# Cleanup
module purge

