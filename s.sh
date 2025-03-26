#!/bin/bash
#SBATCH --job-name="[BabyLM] Pretraining loris3/babylm_2024_10m_curriculum, roberta, random.pt"
#SBATCH --container-image=loris3/cuda:latest
#SBATCH --container-mount-home 
#SBATCH --mem=128G 
#SBATCH --cpus-per-task=24  
#SBATCH --gres=gpu:4
#SBATCH --time=0-12:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/babylm
#SBATCH --nodelist=dgx-h100-em2


python3 --version

python3 pretrain.py loris3/babylm_2024_10m_curriculum random.pt

