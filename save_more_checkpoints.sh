#!/bin/bash
#SBATCH --job-name="[BabyLM] Best Model Additional Checkpoints"
#SBATCH --container-image=loris3/babylm:latest
#SBATCH --container-mount-home 
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24  
#SBATCH --gres=gpu:4
#SBATCH --time=0-05:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/babylm
#SBATCH --nodelist=galadriel,dgx-h100-em2
#SBATCH --nodes=1

python3 --version


python3 pretrain.py loris3/stratified_10m_curriculum roberta_influence_incr_bins_lognorm.pt --model_type=roberta --more_checkpoints

