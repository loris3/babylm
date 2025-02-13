import argparse
import subprocess
import os
def main():
    print(os.getcwd())
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for gradient extraction and influence computation.")
    parser.add_argument("model", type=str, help="model name.")
    parser.add_argument("dataset_train", type=str, help="train dataset name.")
    parser.add_argument("dataset_train_split", type=str, help="train split name")

    parser.add_argument("dataset_test", type=str, help="test dataset name.")
    parser.add_argument("dataset_test_split", type=str, help="test split name")

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

    dataset_test_split  = args.dataset_test_split + "[0:100%]" if not "[" in args.dataset_test_split else args.dataset_test_split

    prev_job_ids_gradients = []

    checkpoint_ids = None

    # hotfix: N+1 checkpoints are created for equitoken datasets, skip the first/include the final model 
    if "llama" in args.dataset_train:
        checkpoint_ids = list(range(1, args.n_checkpoints+1)) if args.checkpoints is None else args.checkpoints

    else:
        checkpoint_ids = list(range(args.n_checkpoints)) if args.checkpoints is None else args.checkpoints

    for i in checkpoint_ids:


        # first, extract gradients for test dataset (used troughout all superbatches)
        
        if len(prev_job_ids_gradients) > 0: # add dependency: influence computation for last checkpoint is finished
            dependency = f"--dependency=afterany:{prev_job_ids_gradients[-1]}"
        else:
            dependency = ""

        extract_command = [
            "sbatch",
            f"--job-name=test set gradient extraction for checkpoint {i}",
            dependency,
            "./slurm_extract_gradients.sh",
            args.model,
            args.dataset_test,
            str(i),
            dataset_test_split,
            args.paradigm
        ]

        extract_command = [c for c in extract_command if c != ""]
        if args.debug:
            extract_command_str = " ".join([c for c in extract_command])
            print(f"[DEBUG] {extract_command_str}")
            test_gradients_job_id = f"job_{i}_test_extract"  # Mock job ID in args.debug mode
        else:
            extract_process = subprocess.run(extract_command, stdout=subprocess.PIPE, text=True, check=True)
            test_gradients_job_id = extract_process.stdout.strip().split()[-1] # get SLURM job ID 




        assert 100 % args.superbatches == 0
      
        for train_dataset_split in [args.dataset_train_split + f"[{i}:{i + 100 // args.superbatches}%]" for i in range(0, 100, 100 // args.superbatches)]:
            
            # gradient extraction
            if len(prev_job_ids_gradients) == args.max_concurrent_gradient_extraction_scripts: # add dependency if more than n gradient extraction scripts are scheduled 
                dependency = f"--dependency=afterany:{prev_job_ids_gradients[0]},afterok:{test_gradients_job_id}"
                prev_job_ids_gradients = prev_job_ids_gradients[1:]
            else:
                dependency = f"--dependency=afterok:{test_gradients_job_id}"



            
            extract_command = [
                "sbatch",
                "--nice=10",
                f"--job-name=aio computation for checkpoint {i}: {train_dataset_split}",
                dependency,
                "./slurm_aio.sh",
                args.model,
                args.dataset_train,
                train_dataset_split,
                args.dataset_test,
                dataset_test_split,
                str(i),
                args.paradigm

            ]
            extract_command = [c for c in extract_command if c != ""]
            if args.debug:
                extract_command_str = " ".join([c for c in extract_command])
                print(f"[DEBUG] {extract_command_str}")
                job_id = f"job_{i}_{train_dataset_split}_extract"  # Mock job ID in args.debug mode
            else:
                extract_process = subprocess.run(extract_command, stdout=subprocess.PIPE, text=True, check=True)
                job_id = extract_process.stdout.strip().split()[-1] # get SLURM job ID 

            prev_job_ids_gradients.append(job_id)

          
       
if __name__ == "__main__":
    main()
