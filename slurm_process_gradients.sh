#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --singleton
#SBATCH --job-name="training data influence calculation: influence computation"
#SBATCH --comment="RoBERTa Training Data Influence Experiments"
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=650G
#SBATCH --cpus-per-task=36
source /etc/profile.d/modules.sh

# Create a permanent environment with the name "my_new_permanent_environment"
export ENV_MODE="permanent"
export ENV_NAME="bayblm"
# This environment is now stored in $HOME/venvs/my_new_permanent_environment
 
# Load module miniforge3
module load miniforge

# cp -r /srv/home/users/loriss21cs/babylm/gradients $TMPDIR/gradients
# df -h $TMPDIR
# ls $TMPDIR/gradients



conda env update --file environment.yml
conda activate 
# Run your python script
python process_gradients.py $2 $3 $1
 
# Cleanup
module purge

