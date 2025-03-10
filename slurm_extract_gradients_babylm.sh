#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="gradiend extraction"
#SBATCH --comment="Gradient extraction for training data influence experiments."
#SBATCH --time=0-08:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=24
#SBATCH --nodelist=galadriel,dgx-h100-em2,dgx1
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



# conda env update --file environment.yaml

# Run your python script
python extract_gradients.py $1 $2 $3 --dataset_split=$4 --paradigm=$5 --mode=store --num_processes_gradients=20 --skip_if_gradient_folder_exists --gradients_per_file=1000



# Cleanup
module purge

