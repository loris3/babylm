#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="cleanup"
#SBATCH --comment="Deletes gradients stored on network share"
#SBATCH --time=0-00:20:00
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=dgx-h100-em2,dgx1,galadriel
source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="babylm_venv"
module load miniforge
checkpoint_name=$(python get_checkpoint_name.py $1 $2)


rm -r "./gradients/$3/$4/$5/$checkpoint_name/"*

echo deleted "./gradients/$3/$4/$5/$checkpoint_name/"*

module purge