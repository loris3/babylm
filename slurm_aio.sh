#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="gradiend extraction"
#SBATCH --comment="RoBERTa Training Data Influence Experiments."
#SBATCH --time=0-06:00:00
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --mem=874GB
#SBATCH --cpus-per-task=24
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



# conda env update --file environment.yaml

# Run your python script
python extract_gradients.py $1 $2 $6 --dataset_split=$3 --paradigm=$7 --gradients_output_path=$TMPDIR
python process_gradients.py $1 $2 $6 --dataset_train_split=$3 --dataset_test=$4 --dataset_test_split=$5 --mode=mean --gradients_output_path=$TMPDIR

# Cleanup
module purge

