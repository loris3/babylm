import argparse
import subprocess
import os
def main():
    print(os.getcwd())
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for gradient extraction and influence computation.")
    parser.add_argument("model", type=str, help="model name.")
    parser.add_argument("dataset", type=str, help="dataset name.")
    parser.add_argument("--max_concurrent_gradient_extraction_scripts",help="Maximum number of gradient extraction scripts to have running at a time. Adds SLURM dependencies", type=int, default=2)
    parser.add_argument("--max_concurrent_influence_computation_scripts",help="Maximum number of influence computation scripts to have running at a time. Adds SLURM dependencies", type=int, default=2)
    parser.add_argument("--debug", action="store_true", help="Log commands instead of executing them.")


    class SplitArgs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, [int(i) for i in values.split(",")])

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_checkpoints",help="Number of checkpoints to process, starting at ID 0", type=int)
    group.add_argument("--checkpoints",action=SplitArgs, help="Comma seperated checkpoint IDs, starting at 0, (e.g., '0,1,5,9')" )
    
    args = parser.parse_args()


    prev_job_ids_gradients = []
    prev_job_ids_influence = []

    checkpoint_ids = list(range(args.n_checkpoints)) if args.checkpoints is None else args.checkpoints

    for i in checkpoint_ids:

        # add dependency if more than n gradient extraction scripts are scheduled 
        if len(prev_job_ids_gradients) == args.max_concurrent_gradient_extraction_scripts:
            dependency = f"--dependency=afterany:{prev_job_ids_gradients[0]}"
            prev_job_ids_gradients = prev_job_ids_gradients[1:]
        else:
            dependency = ""

        # gradient extraction
        extract_command = [
            "sbatch",
            f"--job-name=gradient extraction for checkpoint {i}",
            dependency,
            "./slurm_extract_gradients.sh",
            args.model,
            args.dataset,
            str(i)
        ]
        extract_command = [c for c in extract_command if c != ""]
        if args.debug:
            extract_command_str = " ".join([c for c in extract_command])
            print(f"[DEBUG] {extract_command_str}")
            job_id = f"job_{i}_extract"  # Mock job ID in args.debug mode
        else:
            extract_process = subprocess.run(extract_command, stdout=subprocess.PIPE, text=True, check=True)
            job_id = extract_process.stdout.strip().split()[-1] # get SLURM job ID 

        prev_job_ids_gradients.append(job_id)

        ##############
        # influence computation (dependent)
        # add dependency if more than n influence compuation scripts are scheduled 
        if len(prev_job_ids_influence) == args.max_concurrent_influence_computation_scripts:
            dependency = f"--dependency=afterany:{prev_job_ids_influence[0]},afterok:{job_id}"
            prev_job_ids_influence = prev_job_ids_influence[1:]
        else:
            dependency = f"--dependency=afterok:{job_id}"

      
        influence_command = [
            "sbatch",
            "--nice=10",
            f"--job-name=influence computation for checkpoint {i} ",
            dependency,
            "./slurm_process_gradients.sh",
            args.model,
            args.dataset,
            str(i)
        ]
        influence_command = [c for c in influence_command if c != ""]
        if args.debug:
            influence_command_str = " ".join([c for c in influence_command])
            print(f"[DEBUG] {influence_command_str}")
            job_id = f"job_{i}_influence"  # Mock job ID in args.debug mode
        else:
            influence_process = subprocess.run(influence_command, stdout=subprocess.PIPE, text=True, check=True)
            job_id = influence_process.stdout.strip().split()[-1] # get SLURM job ID 

        prev_job_ids_influence.append(job_id)
        ##################
        # influence_command = [
        #     "sbatch",
        #     f"--job-name=influence computation for checkpoint {i}",
        #     f"--dependency=afterok:{job_id}",
        #     "./slurm_process_gradients.sh",
        #     args.model,
        #     args.dataset,
        #     str(i)
        # ]
        # influence_command_str = " ".join(influence_command)
        # if args.debug:
        #     print(f"[DEBUG] {influence_command_str}")
        # else:
        #     subprocess.run(influence_command, check=True)

if __name__ == "__main__":
    main()
