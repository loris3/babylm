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



    parser.add_argument("--max_precompute_superbatches",help="Maximum number of gradient extraction tasks to run before processing. Adds SLURM dependencies", type=int, default=2)
    parser.add_argument("--debug", action="store_true", help="Log commands instead of executing them.")
    parser.add_argument("--single_influence_job", action="store_true", help="Uses one job for all test sets: enables caching of gradients to the node's /tmp partition. If omitted, one job per test set is created") 
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
                f"--job-name=Extracting gradients for {test_dataset_name} {test_dataset_split} (checkpoint {i})",
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
        cleanup_job_ids = []
        assert 100 % args.superbatches == 0
        for train_dataset_split in [args.dataset_train_split + f"[{i}%:{i + 100 // args.superbatches}%]" for i in range(0, 100, 100 // args.superbatches)]:
            
            # gradient extraction for superbatch
            dependency = f"--dependency=afterok:{test_gradients_job_id}"  # at most max_precompute_superbatches gradient extraction jobs
            if len(cleanup_job_ids) >= args.max_precompute_superbatches:
                dependency += ":"+cleanup_job_ids[0]
                cleanup_job_ids = cleanup_job_ids[args.max_precompute_superbatches:]

            extract_command = [
                "sbatch",
                f"--job-name=Extracting gradients for {train_dataset_split} (checkpoint {i})",
                dependency,
                "./slurm_extract_gradients.sh",
                args.model,
                args.dataset_train,
                str(i),
                train_dataset_split,
                args.paradigm,
                "store",

            ]
            extract_command = [c for c in extract_command if c != ""]
            if args.debug:
                extract_command_str = " ".join([c for c in extract_command])
                print(f"[DEBUG] {extract_command_str}")
                job_id_gradients_train = f"job_{i}_{train_dataset_split}_extract"  # Mock job ID in args.debug mode
            else:
                extract_process = subprocess.run(extract_command, stdout=subprocess.PIPE, text=True, check=True)
                job_id_gradients_train = extract_process.stdout.strip().split()[-1] # get SLURM job ID 


            ##############
            # influence computation
            # per test dataset
            # per superbatch
            import shutil

            job_ids_test_set_influence = []

            if args.single_influence_job:
                # create one influence estimation job so that subsequent test sets re-use gradients cached on /tmp
                
                # copy template
                shutil.copyfile("./slurm_process_gradients_template.sh", "./slurm_process_gradients_combined.sh")

                with open("./slurm_process_gradients_combined.sh", 'a') as fd:
                    # append one python extract_gradients.py call per test set
                    for test_dataset_name, test_dataset_split in test_datasets:
                        c = f"python process_gradients.py {args.model} {args.dataset_train} {i} --dataset_train_split={train_dataset_split} --dataset_test={test_dataset_name} --dataset_test_split={test_dataset_split} --mode=mean  --batch_size=10"
                        fd.write(f'\n{c}')


                dependency = f"--dependency=afterok:{job_id_gradients_train}:{':'.join(job_ids_test_sets)}" # run after gradient scripts for test sets and superbatch are complete
                influence_command = [
                    "sbatch",
                    "--nice=10",
                    f"--job-name=Calculation of training data influence for {train_dataset_split} (checkpoint {i})",
                    dependency,
                    "./slurm_process_gradients_combined.sh",
                ]
                influence_command = [c for c in influence_command if c != ""]
                influence_command_str = " ".join([c for c in influence_command])
                if args.debug:
                    
                    print(f"[DEBUG] {influence_command_str}")
                    job_id_influence = f"job_{i}_{train_dataset_split}_influence_{test_dataset_name}{test_dataset_split}"  # Mock job ID in args.debug mode
                else:
                    influence_process = subprocess.run(influence_command, stdout=subprocess.PIPE, text=True, check=True)
                    job_id_influence = influence_process.stdout.strip().split()[-1] # get SLURM job ID 

                job_ids_test_set_influence.append(job_id_influence)
            else:
                    # create multiple influence estimation jobs, each one will load gradients independently from network share

                    for test_dataset_name, test_dataset_split in test_datasets:
                        dependency = f"--dependency=afterok:{job_id_gradients_train}:{':'.join(job_ids_test_sets)}"
                        influence_command = [
                            "sbatch",
                            "--nice=10",
                            f"--job-name=Calculation of training data influence between {train_dataset_split} and {test_dataset_name} (checkpoint {i})",
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
                            job_id_influence = f"job_{i}_{train_dataset_split}_influence_{test_dataset_name}{test_dataset_split}"  # Mock job ID in args.debug mode
                        else:
                            influence_process = subprocess.run(influence_command, stdout=subprocess.PIPE, text=True, check=True)
                            job_id_influence = influence_process.stdout.strip().split()[-1] # get SLURM job ID 

                    job_ids_test_set_influence.append(job_id_influence)





            # cleanup after superbatch
            dependency = f"--dependency=afterok:{':'.join(job_ids_test_set_influence)}"
            cleanup_command = [
                "sbatch",
                "--nice=10",
                f"--job-name=Deleting gradients for {train_dataset_split}",
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
            cleanup_job_ids.append(cleanup_job_id)
            
       
       
if __name__ == "__main__":
    main()
