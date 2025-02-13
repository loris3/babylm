#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --job-name="gradiend extraction"
#SBATCH --comment="RoBERTa Training Data Influence Experiments."
#SBATCH --time=0-00:20:00
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=dgx-h100-em2,dgx1
source /etc/profile.d/modules.sh


checkpoint_name=$(python get_checkpoint_name.py $1 $2)


rm -rf ./gradients/$3/$4/$5/$checkpoint_name
 

