#!/bin/bash 
#SBATCH --job-name='[Yuxi: Verbalized Confidence] Email when done'

#SBATCH --mem=1GB 
#SBATCH --cpus-per-task=1  
#SBATCH --time=0-00:01:00
#SBATCH --nodes=1
#SBATCH --dependency=afterany:55907:55903
#SBATCH --nodelist=dgx1,dgx-h100-em2,galadriel
#SBATCH --mail-type=END
#SBATCH --mail-user=yuxi.xia@univie.ac.at
echo "Jobs complete"