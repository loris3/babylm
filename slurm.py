import argparse
import subprocess
import os
import shutil


CONTAINER_IMAGE = "loris3/babylm:latest"
NODELIST_EXTRACT = "dgx-h100-em2"
NODELIST_PROCESS = "dgx-h100-em2"


MEM_EXTRACT = "64GB"
MEM_PROCESS = "128GB"
TIME_EXTRACT_TEST ="0-19:00:00"

TIME_EXTRACT_TRAIN_PER_SUPERCHUNK ="0-19:00:00"
TIME_PROCESS ="0-12:00:00"

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

def main():
    print(os.getcwd())
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for gradient extraction and influence computation.")
    parser.add_argument("model", type=str, help="model name.")
    parser.add_argument("dataset_train", type=str, help="train dataset name.")
    parser.add_argument("dataset_train_split", type=str, help="train split name")
    parser.add_argument("--proj_dim", type=int, nargs="?", const=1, default=2**14)

    parser.add_argument("--test_datasets", type=str, nargs="+", help="List of dataset_name split_name pairs.")
        
    parser.add_argument("--stop_after_first", action="store_true", help="Only run for first superbatch")
    parser.add_argument("--gradients_per_file", help="Number of gradients per output file", type=int, nargs="?", const=1, default=1000) # 10000 = ~7.4 GB per file for BERT

    parser.add_argument("--debug", action="store_true", help="Log commands instead of executing them.")
    parser.add_argument("--superbatches",help="Times the training dataset should be split", type=int, default=1)
    parser.add_argument("--paradigm", help="One of 'mlm', 'pre' or 'sft'", default="pre")
    parser.add_argument("--mode", help="One of 'single', 'mean', 'mean_normalized'.If 'mean', mean influence of individual train on all examples in test; if 'single' 1 train -> 1 test", default="single")
    parser.add_argument("--random_projection", default=False, action='store_true')
    parser.add_argument("--store", default=False, action='store_true')
    class SplitArgs(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, [int(i) for i in values.split(",")])

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_checkpoints",help="Number of checkpoints to process, starting at ID 0", type=int)
    group.add_argument("--checkpoints",action=SplitArgs, help="Comma seperated checkpoint IDs, starting at 0, (e.g., '0,1,5,9')")
    
    
    args = parser.parse_args()

    assert args.store or ("mean" in args.mode), "Check args: nothing is stored"

    test_datasets = [(args.test_datasets[i], args.test_datasets[i+1] + "[0%:100%]" if not "[" in args.test_datasets[i+1] else args.test_datasets[i+1]) for i in range(0, len(args.test_datasets), 2)]


    


    checkpoint_ids = None

    # hotfix: N+1 checkpoints are created for equitoken datasets, skip the first/include the final model 
    if "llama" in args.dataset_train:
        checkpoint_ids = list(range(1, args.n_checkpoints+1)) if args.checkpoints is None else args.checkpoints

    else:
        checkpoint_ids = list(range(args.n_checkpoints)) if args.checkpoints is None else args.checkpoints

    for checkpoint_id in checkpoint_ids:
        job_ids_test_sets = []
        # first, extract gradients for test datasets (re-used troughout all superbatches)
        
        for test_dataset_name, test_dataset_split in test_datasets:
          
         
            extract_script_test = \
f"""
#!/bin/bash
#SBATCH --job-name="[IE Experiments] Extracting Gradients {test_dataset_name} {test_dataset_split} (checkpoint {checkpoint_id})"
#SBATCH --container-image={CONTAINER_IMAGE}
#SBATCH --container-mount-home 
#SBATCH --mem={MEM_EXTRACT} 
#SBATCH --cpus-per-task=12  
#SBATCH --gres=gpu:1
#SBATCH --time={TIME_EXTRACT_TEST}
#SBATCH --container-workdir={os.getcwd()}
#SBATCH --nodelist={NODELIST_EXTRACT}
#SBATCH --nice
#SBATCH --nodes=1

python3 --version

df -h

python3 extract_gradients.py \
{args.model} \
{test_dataset_name} \
{checkpoint_id} \
--dataset_split={test_dataset_split} \
--paradigm={args.paradigm} \
--gradients_per_file={args.gradients_per_file} \
--mode={args.mode} \
{"--random_projection" if args.random_projection else ""} \
{"--store" if args.store else ""}
"""


            
          
            test_gradients_job_id = submit_script(extract_script_test, args, debug_id=f"job_{checkpoint_id}_test_extract")
            job_ids_test_sets.append(test_gradients_job_id)




        # now iterate over the train dataset in "superbatches" and compute influence for each test dataset
        cleanup_job_ids = []
        assert 100 % args.superbatches == 0
        for train_dataset_split in [args.dataset_train_split + f"[{i}%:{i + 100 // args.superbatches}%]" for i in range(0, 100, 100 // args.superbatches)]:

            # if train_dataset_split not in ["train[17%:18%]","train[18%:19%]"]:
            #     print("skipping", train_dataset_split)
            #     continue
            job_id_gradients_train = None


            if (test_dataset_name != args.dataset_train) or  (train_dataset_split  != test_dataset_split):
                extract_script_train = \
            f"""
            #!/bin/bash
            #SBATCH --job-name="[IE Experiments] Extracting Gradients {train_dataset_split} (checkpoint {checkpoint_id})"
            #SBATCH --container-image={CONTAINER_IMAGE}
            #SBATCH --container-mount-home 
            #SBATCH --mem={MEM_EXTRACT} 
            #SBATCH --cpus-per-task=24  
            #SBATCH --gres=gpu:1
            #SBATCH --time={TIME_EXTRACT_TRAIN_PER_SUPERCHUNK}
            #SBATCH --container-workdir={os.getcwd()}
            #SBATCH --nodelist={NODELIST_EXTRACT}
            #SBATCH --dependency=afterok:51655{":".join([str(i) for i in job_ids_test_sets])}
            #SBATCH --nodes=1

            df -h

            python3 --version

            python3 extract_gradients.py \
            {args.model} \
            {args.dataset_train} \
            {checkpoint_id} \
            --dataset_split={train_dataset_split} \
            --paradigm={args.paradigm} \
            --gradients_per_file={args.gradients_per_file} \
            --mode=store \
            {"--random_projection" if args.random_projection else ""} \
            {"--store" if args.store else ""}
            """


          
                job_id_gradients_train = submit_script(extract_script_train, args, debug_id=f"job_{checkpoint_id}_{train_dataset_split}_extract")
            else:
                print("fff skipping train")
            ##############
            # influence computation
            # per test dataset
            # per superbatch
            

            job_ids_processing = []
            
     
                # create one influence estimation job so that subsequent test sets re-use gradients cached on /tmp
            dependency = f"{job_id_gradients_train}:{':'.join(job_ids_test_sets)}" if job_id_gradients_train is not None else f"{':'.join(job_ids_test_sets)}"
            process_script = \
f"""
#!/bin/bash
#SBATCH --job-name="[IE Experiments] Processing Gradients {train_dataset_split} (checkpoint {checkpoint_id})"
#SBATCH --container-image={CONTAINER_IMAGE}
#SBATCH --container-mount-home 
#SBATCH --mem={MEM_PROCESS} 
#SBATCH --cpus-per-task=24  
#SBATCH --time={TIME_PROCESS}
#SBATCH --container-workdir={os.getcwd()}
#SBATCH --nodelist={NODELIST_PROCESS}
#SBATCH --dependency={dependency}
#SBATCH --nodes=1

df -h
python3 --version

"""


          
            # append one python extract_gradients.py call per test set
            for test_dataset_name, test_dataset_split in test_datasets:
                c = \
f"""
python3 process_gradients.py \
    {args.model} \
    {args.dataset_train} \
    {checkpoint_id} \
    --dataset_train_split={train_dataset_split} \
    --dataset_test={test_dataset_name} \
    --dataset_test_split={test_dataset_split} \
    --gradients_per_file={args.gradients_per_file} \
    --mode={args.mode} \
    {"--random_projection" if args.random_projection else ""} \
    --batch_size=10
    """
                process_script += f'\n{c}'


            job_id_processing = submit_script(process_script, args, debug_id=f"job_{checkpoint_id}_{train_dataset_split}_influence_{test_dataset_name}{test_dataset_split}")
            job_ids_processing.append(job_id_processing)
    
            if args.stop_after_first:
                print("[DEBUG] stopping after first superbatch")
                exit()
        # cleanup after superbatch
        cleanup_script = \
f"""
#!/bin/bash
#SBATCH --job-name="Cleanup for {train_dataset_split} (checkpoint {checkpoint_id})"
#SBATCH --comment="Deletes gradients stored on network share"
#SBATCH --time=0-00:20:00
#SBATCH --container-image={CONTAINER_IMAGE}
#SBATCH --container-mount-home 
#SBATCH --mem={MEM_PROCESS} 
#SBATCH --cpus-per-task=1  
#SBATCH --container-workdir={os.getcwd()}
#SBATCH --nodelist={NODELIST_PROCESS}
#SBATCH --nodes=1

#SBATCH --dependency=afterok:{':'.join(job_ids_processing)}

checkpoint_name=$(python3 get_checkpoint_name.py {args.model} {str(checkpoint_id)})


# rm -r ./gradients/{args.proj_dim}/{os.path.basename(args.model)}/{os.path.basename(args.dataset_train)}/{train_dataset_split}/$checkpoint_name/*

echo deleted "./gradients/{args.proj_dim}/{os.path.basename(args.model)}/{os.path.basename(args.dataset_train)}/{train_dataset_split}/$checkpoint_name/*"

module purge
"""
        cleanup_job_id = submit_script(cleanup_script, args, debug_id= f"job_{checkpoint_id}_{train_dataset_split}_cleanup")
        cleanup_job_ids.append(cleanup_job_id)

        
   
       
if __name__ == "__main__":
    main()
