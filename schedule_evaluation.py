import argparse
import subprocess
import os
import shutil



import config




from itertools import product

from slurm_utils import submit_script

CONTAINER_IMAGE = "loris3/babylm:eval"
NODELIST = "dgx-h100-em2,galadriel"




MEM = "32GB"
TIME ="0-2:00:00"



def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for gradient extraction and influence computation.")
    parser.add_argument("--debug", action="store_true", help="Log commands instead of executing them.")

    
    args = parser.parse_args()

    get_model_name = lambda dataset, model_type, curriculum: os.path.join(dataset + ("_" + model_type) + "_" + curriculum.split(".")[0])


    jobs =[(get_model_name(dataset, model_type, curriculum), dataset, model_type, curriculum) for dataset, model_type, curriculum in product(config.datasets, config.model_types, config.baseline_curricula)]
    jobs.extend([(get_model_name(dataset, model_type, model_type + curriculum), dataset, model_type, model_type + curriculum) for dataset, model_type, curriculum  in (product(config.datasets, config.model_types, config.influence_curricula))])
    jobs.extend([(model_name, "external", model_type, "external") for model_name, model_type in config.baseline_models])

    for model, dataset, model_type, curriculum in jobs:
       
        if not os.path.exists(os.path.join("./models", os.path.basename(model))) and dataset != "external":
            print("skipping", model, "not ready")
            continue

        script_header = \
f"""
#!/bin/bash 
#SBATCH --job-name='[BabyLM] (BLIMP) Evaluation {model}'
#SBATCH --container-image={CONTAINER_IMAGE}
#SBATCH --container-mount-home 
#SBATCH --mem={MEM} 
#SBATCH --cpus-per-task=24  

#SBATCH --time={TIME}
#SBATCH --nodes=1
#SBATCH --container-workdir={os.getcwd()}
#SBATCH --nodelist={NODELIST}

export $(grep -v '^#' .env | xargs) && huggingface-cli login --token $HF_TOKEN
python3 --version


"""
        blimp_out_path = f"./eval/blimp/{os.path.basename(model)}/blimp_results.json"
        blimp = \
f"""

python3 -m lm_eval --model hf{"-mlm" if model_type == "roberta" else ""} \
    --model_args pretrained={model},trust_remote_code=True,backend='{"mlm" if model_type == "roberta" else "causal"}' \
    --tasks blimp_filtered,blimp_supplement \
    --device auto \
    --batch_size 1 \
    --log_samples \
    --output_path {blimp_out_path}

""" 



        if not os.path.isfile(blimp_out_path):
            submit_script(script_header + blimp, args,  debug_id=None)
        else:
            print("skipping", model, "results exist")

        

       
if __name__ == "__main__":
    main()
