#!/bin/bash
#SBATCH --job-name="[IE Anastasiia] Derive subset from pre-computed gradients"
#SBATCH --container-image=loris3/babylm:latest
#SBATCH --container-mount-home 
#SBATCH --mem=128GB 
#SBATCH --cpus-per-task=12  
#SBATCH --time=1-19:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/babylm
#SBATCH --nodelist=galadriel,dgx-h100-em2,dgx1
#SBATCH --nice
#SBATCH --nodes=1

python3 --version
df -h

python3 derive_subset.py allenai/OLMo-2-1124-7B-SFT anasedova/tulu_3_whole_updated anasedova/tulu_3_whole_updated 0  --dataset_subset_split=train[0%:5%] --gradients_per_file=1000 --random_projection 

