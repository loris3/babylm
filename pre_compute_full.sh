#!/bin/bash

for i in {0..9}; do
  start=$((i * 10))
  end=$(((i + 1) * 10))
  sbatch --export=ALL,start=$start,end=$end <<EOF
#!/bin/bash
#SBATCH --job-name="[IE Anastasiia] (Base model) Pre-computing Gradients anasedova/tulu_3_whole_updated train[${start}%:${end}%]"
#SBATCH --container-image=loris3/babylm:latest
#SBATCH --container-mount-home 
#SBATCH --mem=64GB 
#SBATCH --cpus-per-task=12  
#SBATCH --gres=gpu:1
#SBATCH --time=1-19:00:00
#SBATCH --container-workdir=/srv/home/users/loriss21cs/babylm
#SBATCH --nodelist=galadriel,dgx-h100-em2
#SBATCH --nice
#SBATCH --nodes=1

echo "Running for range: ${start}% to ${end}%"
python3 --version
df -h

python3 extract_gradients.py allenai/OLMo-2-1124-7B anasedova/tulu_3_whole_updated 0 --dataset_split=train[${start}%:${end}%] --paradigm=pre --gradients_per_file=1000 --mode=store --random_projection --store
EOF
done
