import argparse
import subprocess
import os
def main():
    print(os.getcwd())
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for gradient extraction and influence computation.")
    parser.add_argument("model", type=str, help="model name.")
    parser.add_argument("dataset_train", type=str, help="train dataset name.")
    parser.add_argument("dataset_train_split", type=str, help="train split name")

    parser.add_argument("--test_datasets", type=str, nargs="+", help="List of dataset_name split_name pairs.")



    parser.add_argument("--max_concurrent_gradient_extraction_scripts",help="Maximum number of gradient extraction scripts to have running at a time. Adds SLURM dependencies", type=int, default=2)
    parser.add_argument("--max_concurrent_influence_computation_scripts",help="Maximum number of influence computation scripts to have running at a time. Adds SLURM dependencies", type=int, default=2)
    parser.add_argument("--debug", action="store_true", help="Log commands instead of executing them.")
        
    parser.add_argument("--superbatches",help="Times the training dataset should be split", type=int, default=1)
    parser.add_argument("--paradigm",help="One of 'mlm', 'pre", default="pre")

    class SplitArgs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, [int(i) for i in values.split(",")])

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_checkpoints",help="Number of checkpoints to process, starting at ID 0", type=int)
    group.add_argument("--checkpoints",action=SplitArgs, help="Comma seperated checkpoint IDs, starting at 0, (e.g., '0,1,5,9')")
    
    args = parser.parse_args()



    test_datasets = [(args.test_datasets[i], args.test_datasets[i+1] + "[0%:100%]" if not "[" in args.test_datasets[i+1] else args.test_datasets[i+1]) for i in range(0, len(args.test_datasets), 2)]


    job_ids_test_sets = []

    prev_job_ids_gpu_res = []
    checkpoint_ids = None

    # hotfix: N+1 checkpoints are created for equitoken datasets, skip the first/include the final model 
    if "llama" in args.dataset_train:
        checkpoint_ids = list(range(1, args.n_checkpoints+1)) if args.checkpoints is None else args.checkpoints

    else:
        checkpoint_ids = list(range(args.n_checkpoints)) if args.checkpoints is None else args.checkpoints

    for i in checkpoint_ids:


        # first, extract gradients for test datasets (used troughout all superbatches)
        
        for test_dataset_name, test_dataset_split in test_datasets:
            dependency = ""
            extract_command = [
                "sbatch",
                f"--job-name=test set {test_dataset_name} {test_dataset_split} gradient extraction for checkpoint {i}",
                dependency,
                "./slurm_extract_gradients.sh",
                args.model,
                test_dataset_name,
                str(i),
                test_dataset_split,
                args.paradigm,
                "store_mean"
            ]

            extract_command = [c for c in extract_command if c != ""]
            if args.debug:
                extract_command_str = " ".join([c for c in extract_command])
                print(f"[DEBUG] {extract_command_str}")
                test_gradients_job_id = f"job_{i}_test_extract"  # Mock job ID in args.debug mode
            else:
                extract_process = subprocess.run(extract_command, stdout=subprocess.PIPE, text=True, check=True)
                test_gradients_job_id = extract_process.stdout.strip().split()[-1] # get SLURM job ID 
            job_ids_test_sets.append(test_gradients_job_id)




        # now iterate over the train dataset in "superbatches" and compute influence for each test dataset
        cleanup_job_id = None
        assert 100 % args.superbatches == 0
        for train_dataset_split in [args.dataset_train_split + f"[{i}%:{i + 100 // args.superbatches}%]" for i in range(0, 100, 100 // args.superbatches)]:
            # gradient extraction
            if len(prev_job_ids_gpu_res) == args.max_concurrent_gradient_extraction_scripts: # add dependency if more than n gradient extraction scripts are scheduled 
                dependency = f"--dependency=afterany:{prev_job_ids_gpu_res[0]},afterok:{':'.join(job_ids_test_sets)}" + (":"+cleanup_job_id if cleanup_job_id is not None else "")
                prev_job_ids_gpu_res = prev_job_ids_gpu_res[1:]
            else:
                dependency = f"--dependency=afterok:{test_gradients_job_id}"  + (":"+cleanup_job_id if cleanup_job_id is not None else "")
            
            extract_command = [
                "sbatch",
                f"--job-name={train_dataset_split} gradient extraction train for checkpoint {i}",
                dependency,
                "./slurm_extract_gradients.sh",
                args.model,
                args.dataset_train,
                str(i),
                train_dataset_split,
                args.paradigm,
                "store"
            ]
            extract_command = [c for c in extract_command if c != ""]
            if args.debug:
                extract_command_str = " ".join([c for c in extract_command])
                print(f"[DEBUG] {extract_command_str}")
                job_id = f"job_{i}_{train_dataset_split}_extract"  # Mock job ID in args.debug mode
            else:
                extract_process = subprocess.run(extract_command, stdout=subprocess.PIPE, text=True, check=True)
                job_id = extract_process.stdout.strip().split()[-1] # get SLURM job ID 

            prev_job_ids_gpu_res.append(job_id)

            ##############
            # influence computation (dependent)
            # per test dataset
            # per superbatch
            job_ids_test_set_influence = []
            for test_dataset_name, test_dataset_split in test_datasets:
                dependency = f"--dependency=afterok:{job_id}:{':'.join(job_ids_test_sets)}"
                influence_command = [
                    "sbatch",
                    "--nice=10",
                    f"--job-name=influence computation for checkpoint {i} ",
                    dependency,
                    "./slurm_process_gradients.sh",
                    args.model,
                    args.dataset_train,
                    train_dataset_split,
                    test_dataset_name,
                    test_dataset_split,
                    str(i)
                ]
                influence_command = [c for c in influence_command if c != ""]
                if args.debug:
                    influence_command_str = " ".join([c for c in influence_command])
                    print(f"[DEBUG] {influence_command_str}")
                    job_id = f"job_{i}_{train_dataset_split}_influence_{test_dataset_name}{test_dataset_split}"  # Mock job ID in args.debug mode
                else:
                    influence_process = subprocess.run(influence_command, stdout=subprocess.PIPE, text=True, check=True)
                    job_id = influence_process.stdout.strip().split()[-1] # get SLURM job ID 

                job_ids_test_set_influence.append(job_id)

            # cleanup after superbatch
            dependency = f"--dependency=afterok:{':'.join(job_ids_test_set_influence)}"
            cleanup_command = [
                "sbatch",
                "--nice=10",
                f"--job-name=cleanup for batch {args.dataset_train}{train_dataset_split} ",
                dependency,
                "./slurm_cleanup.sh",
                args.model,
                str(i),
                os.path.basename(args.model),
                os.path.basename(args.dataset_train),
                train_dataset_split,
              
            ]
            cleanup_command = [c for c in cleanup_command if c != ""]
            if args.debug:
                cleanup_command_str = " ".join([c for c in cleanup_command])
                print(f"[DEBUG] {cleanup_command_str}")
                cleanup_job_id = f"job_{i}_{train_dataset_split}_cleanup"  # Mock job ID in args.debug mode
            else:
                cleanup_process = subprocess.run(cleanup_command, stdout=subprocess.PIPE, text=True, check=True)
                cleanup_job_id = cleanup_process.stdout.strip().split()[-1] # get SLURM job ID 
            
            break
       
if __name__ == "__main__":
    main()
