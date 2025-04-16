#!/bin/bash 
#SBATCH --job-name='[BabyLM] Email when done'

#SBATCH --mem=1GB 
#SBATCH --cpus-per-task=1  
#SBATCH --time=0-00:01:00
#SBATCH --nodes=1
#SBATCH --dependency=afterany:55557:55556:55555:55554:55553:55550:55551
#SBATCH --nodelist=dgx1,dgx-h100-em2,galadriel
#SBATCH --mail-type=END
#SBATCH --mail-user=alerts@loris.fyi
echo "Jobs complete"