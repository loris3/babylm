import argparse
import subprocess
import os
import shutil


CONTAINER_IMAGE = "loris3/babylm:latest"
NODELIST = "galadriel,dgx-h100-em2"



MEM = "128G"
TIME ="0-05:00:00"



import config




from slurm_utils import submit_script
from itertools import product
def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for gradient extraction and influence computation.")
    parser.add_argument("stage", help="Either 'baselines' or 'influence'")
    parser.add_argument("--debug", action="store_true", help="Log commands instead of executing them.")

    
    args = parser.parse_args()


    jobs = None
    if args.stage == "baselines":
        jobs = list(product(config.datasets, config.model_types, config.baseline_curricula))  
    else:
        jobs = [(dataset, model_type, model_type + curriculum) for dataset, model_type, curriculum in product(config.datasets, config.model_types, config.influence_curricula)]
        # skip influence curricula jobs with non-exisitng local "random" model folder
        jobs = [job for job in jobs if  os.path.exists(os.path.join("models", os.path.basename(job[0]) + "_" + job[1] + "_" + "random"))]

    # skip jobs with existing local model folder
    jobs = [job for job in jobs if not os.path.exists(os.path.join("models", os.path.basename(job[0]) + "_" + job[1] + "_" + job[2].replace(".pt","")))]
    print(jobs)
    
    for dataset, model_type, curriculum in jobs:
         
        script = \
f"""
#!/bin/bash
#SBATCH --job-name="[BabyLM] Pretraining"
#SBATCH --container-image={CONTAINER_IMAGE}
#SBATCH --container-mount-home 
#SBATCH --mem={MEM} 
#SBATCH --cpus-per-task=24  
#SBATCH --gres=gpu:4
#SBATCH --time={TIME}
#SBATCH --container-workdir={os.getcwd()}
#SBATCH --nodelist={NODELIST}
#SBATCH --nodes=1

python3 --version


python3 pretrain.py {dataset} {curriculum} --model_type={model_type}

""" 

        submit_script(script, args,  debug_id=None)
     

       
if __name__ == "__main__":
    main()
