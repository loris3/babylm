import argparse
import subprocess
import os
import shutil



import config



from itertools import product
def main():
    parser = argparse.ArgumentParser(description="Submit SLURM jobs for gradient extraction and influence computation.")
    parser.add_argument("--debug", action="store_true", help="Log commands instead of executing them.")

    
    args = parser.parse_args()


    jobs = list(product(config.datasets, config.model_types))
    print(len(jobs))

    for dataset, model_type in jobs:
        curriculum = "random.pt"
      

        command = [
            "python3", "slurm.py", 
                os.path.join(dataset + ("_" + model_type) + "_" + curriculum.split(".")[0]),  # model
            dataset,  # dataset_train
            "train",  # dataset_train_split
            "--proj_dim", "16384",
            "--test_datasets", dataset, "train",  
            "--gradients_per_file", "100000",
            "--superbatches", "1",
            "--paradigm", "mlm" if model_type == "roberta" else "pre",
            "--mode", "mean_normalized",
            "--store",
            "--random_projection",
            "--n_checkpoints", "10",
      
        ]

        # Run the script
        result = subprocess.run(command, capture_output=True, text=True)

        # Print output
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)

       
if __name__ == "__main__":
    main()
