import argparse
import subprocess
import os
import shutil


CONTAINER_IMAGE = "loris3/cuda:latest"
NODELIST = "dgx-h100-em2"
NODELIST_PROCESS = "dgx-h100-em2"


MEM = "128G"
TIME ="0-12:00:00"



import config




def submit_script(script, args, debug_id=None):
    if args.debug:
        print(f"[DEBUG] {script}")
        return debug_id
    else:
        with open("s.sh", "w") as script_file:
            script_file.write(script.lstrip("\n"))
        submit_command = ["sbatch", "s.sh"]
        
        extract_process = subprocess.run(submit_command, stdout=subprocess.PIPE, text=True, check=True)
        return extract_process.stdout.strip().split()[-1] # get SLURM job ID 
from itertools import product
def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for gradient extraction and influence computation.")
    parser.add_argument("--debug", action="store_true", help="Log commands instead of executing them.")

    
    args = parser.parse_args()


    jobs = list(product(config.datasets, config.model_types, config.curricula))
    print(len(jobs))

    for dataset, model_type, curriculum in jobs:
         
        script = \
f"""
#!/bin/bash
#SBATCH --job-name="[BabyLM] Pretraining {dataset}, {model_type}, {curriculum}"
#SBATCH --container-image={CONTAINER_IMAGE}
#SBATCH --container-mount-home 
#SBATCH --mem={MEM} 
#SBATCH --cpus-per-task=24  
#SBATCH --gres=gpu:4
#SBATCH --time={TIME}
#SBATCH --container-workdir={os.getcwd()}
#SBATCH --nodelist={NODELIST}


python3 --version

python3 pretrain.py {dataset} {curriculum}

""" 
        submit_script(script, args,  debug_id=None)
        return

       
if __name__ == "__main__":
    main()
