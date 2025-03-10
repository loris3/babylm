#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="influence computation"
#SBATCH --comment="Calculation of training data influence. Loads gradients from network share to /tmp/gradients, assumes /tmp is cleared after job completion"
#SBATCH --time=0-02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=325GB
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



python process_gradients.py allenai/OLMo-2-1124-7B-SFT allenai/tulu-3-sft-olmo-2-mixture 0 --dataset_train_split=train[34%:35%] --dataset_test=anasedova/tulu_3_no_errors --dataset_test_split=train[0%:100%] --mode=mean --batch_size=10
