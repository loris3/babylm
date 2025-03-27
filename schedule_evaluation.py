import argparse
import subprocess
import os
import shutil



import config




from itertools import product

from slurm_utils import submit_script

CONTAINER_IMAGE = "loris3/babylm:eval"
NODELIST = "dgx1"



MEM = "32GB"
TIME ="0-1:00:00"



def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for gradient extraction and influence computation.")
    parser.add_argument("--debug", action="store_true", help="Log commands instead of executing them.")

    
    args = parser.parse_args()


    jobs = list(product(config.datasets, config.model_types, config.curricula))
    print(len(jobs))

    for dataset, model_type, curriculum in jobs:
        model = os.path.join(dataset + ("_" + model_type) + "_" + curriculum.split(".")[0])
        
        script_header = \
f"""
#!/bin/bash 
#SBATCH --job-name='[BabyLM] (BLIMP) Evaluation {model}'
#SBATCH --container-image={CONTAINER_IMAGE}
#SBATCH --container-mount-home 
#SBATCH --mem={MEM} 
#SBATCH --cpus-per-task=24  
#SBATCH --gres=gpu:1
#SBATCH --time={TIME}
#SBATCH --container-workdir={os.getcwd()}
#SBATCH --nodelist={NODELIST}
export $(grep -v '^#' .env | xargs) && huggingface-cli login --token $HF_TOKEN
python3 --version


"""
        blimp_out_path = f"./eval/blimp/{os.path.basename(model)}/blimp_results.json"
        blimp = \
f"""

python3 -m lm_eval --model hf{"-mlm" if model_type == "roberta" else ""} \
    --model_args pretrained={model},backend='{"mlm" if model_type == "roberta" else "causal"}' \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0 \
    --batch_size 1 \
    --log_samples \
    --output_path {blimp_out_path}

""" 



        if not os.path.isfile(blimp_out_path):
            submit_script(script_header + blimp, args,  debug_id=None)

        return

       
if __name__ == "__main__":
    main()
