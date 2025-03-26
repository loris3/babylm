#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="gradiend extraction"
#SBATCH --comment="Gradient extraction for training data influence experiments."
#SBATCH --time=0-02:00:00
#SBATCH --output=test.out

#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=24
#SBATCH --nodelist=galadriel,dgx-h100-em2
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

python extract_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_split=train[0%:1%] --paradigm=sft --mode=store --num_processes_gradients=4 


# Cleanup
module purge

