#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="check skip"
#SBATCH --time=0-00:20:00
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --cpus-per-task=1
#SBATCH --nodelist=dgx-h100-em2,dgx1
source /etc/profile.d/modules.sh
export ENV_MODE="permanent"
export ENV_NAME="babylm_venv"
module load miniforge
ready=$(python check_results_ready.py $1 $2 $6 --dataset_train_split=$3 --dataset_test=$4 --dataset_test_split=$5 --mode=mean  --batch_size=10)
module purge

echo $ready