#!/bin/bash
#SBATCH --job-name="[BabyLM] Waiting for manual file deletion"

#SBATCH --mem=128MB
#SBATCH --cpus-per-task=1  

#SBATCH --time=07-00:00:00

file="/srv/home/groups/dm/share/deleteme_to_launch_experiments"

touch "$file"
chmod 777 "$file"
echo "File created: $file"


while [ -e "$file" ]; do
    sleep 10
done


echo "File has been deleted. Exiting script."
exit 0