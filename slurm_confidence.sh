#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="training data influence when verbalizing confidence"
#SBATCH --comment="Training Data Influence Experiments."
#SBATCH --time=0-10:00:00
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --mem=32G
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
python influence_confidence.py --model=allenai/OLMo-2-1124-7B-SFT --dataset=yuxixia/triviaqa-test-tulu3-query --checkpoint_nr=0 --dataset_split=test[0%:100%] --num_proc=4

# Cleanup
module purge

