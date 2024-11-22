n_checkpoints=$1 
model=$2
dataset=$3

if [ -z "$n_checkpoints" ]; then
    echo "Usage: $0 <n_checkpoints>"
    exit 1
fi

for ((i=0; i<n_checkpoints; i++)); do

    job_id=$(sbatch slurm_extract_gradients.sh $model $dataset $i | awk '{print $4}')
    
    sbatch --dependency=afterok:$job_id slurm_extract_gradients.sh $model $dataset $i
done