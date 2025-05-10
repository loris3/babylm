import argparse
import os

import setproctitle
from dotenv import load_dotenv
load_dotenv()

import argparse
import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset
from util import get_checkpoints_hub
parser = argparse.ArgumentParser("gradient_extraction")

parser.add_argument("model", help="A model on the hf hub. Format: username/name_curriculum")
parser.add_argument("dataset_full", help="A dataset on the hf hub. Format: username/name")

parser.add_argument("dataset_subset", help="A dataset on the hf hub. If supplied, returns one score per test instance. Format: username/name", default=None)
parser.add_argument("--dataset_subset_split", help="The split to access", default="train")

parser.add_argument("checkpoint_nr", help="Id of the checkpoint to extract gradients for (starting at 0)",type=int)
parser.add_argument("--gradients_per_file", help="Number of gradients per output file", type=int, nargs="?", const=1, default=1000)

parser.add_argument("--random_projection", default=False, action='store_true')
parser.add_argument("--proj_dim", type=int, nargs="?", const=1, default=2**14)

args = parser.parse_args()
print("args", args, flush=True)


model_name = args.model.split("/")[-1]
dataset_full_name = args.dataset_full.split("/")[-1]
dataset_full_split_name = "train"
dataset_subset_name = args.dataset_subset.split("/")[-1]
dataset_subset_split_name = args.dataset_subset_split


gradient_output_dir_full = os.path.join("./gradients", str(args.proj_dim) if args.random_projection else "full", model_name, dataset_full_name)
gradient_output_dir_subset = os.path.join("./gradients", str(args.proj_dim) if args.random_projection else "full", model_name, dataset_subset_name, dataset_subset_split_name)


    
if not os.path.exists(gradient_output_dir_subset):
    os.makedirs(gradient_output_dir_subset)

logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')




if __name__ == '__main__':   
    checkpoints =  get_checkpoints_hub(args.model) 
    checkpoint = checkpoints[args.checkpoint_nr]
    
    
    dataset_full = load_dataset(args.dataset_full, split=dataset_full_split_name)
            
    dataset_subset = load_dataset(args.dataset_subset, split=args.dataset_subset_split)

    full_ids = set(dataset_full["ordinal_id"])
    assert len(full_ids) == len(dataset_full), "ordinal_id not unique in full dataset"
    subset_ids = set(dataset_subset["ordinal_id"])
    assert len(subset_ids) == len(dataset_subset), "ordinal_id not unique in subset dataset"

    common_ids = full_ids.intersection(subset_ids)

    matching_indices = [i for i, oid in enumerate(dataset_full['ordinal_id']) if oid in common_ids]

    subdirs = sorted([d for d in os.listdir(gradient_output_dir_full) if d.startswith(dataset_full_split_name)])

    print("subdirs",subdirs)


    gradients = []

    for subdir in tqdm(subdirs, desc="Loading subdirectories"):
        if subdir == dataset_full_split_name:
            continue
        subdir_path = os.path.join(gradient_output_dir_full, subdir, os.path.basename(checkpoint))

        try:
            gradient_files = sorted([f for f in os.listdir(subdir_path) ])
            prin("gradient_files",gradient_files)
    
            for gradient_file in gradient_files:
                file_path = os.path.join(subdir_path, gradient_file)
                g = torch.load(file_path, weights_only=True)
                gradients.append(g)
        except:
            print("skipping", subdir)
            
  
    gradients = torch.cat(gradients, axis=0)
    
    gradients_subset = gradients[matching_indices]
    print("gradients_subset.shape",gradients_subset.shape)
    assert gradients_subset.shape[0] == len(dataset_subset), "gradients_subset.shape[0] != len(dataset_subset)"
    del gradients


    chunks_subset = [ os.path.join(gradient_output_dir_subset, os.path.basename(checkpoint), str(i) + "_" + str(i + args.gradients_per_file)) for i in range(0, gradients_subset.shape[0], args.gradients_per_file)]
    print("chunks_subset",chunks_subset)
    for chunk_path in tqdm(chunks_subset, desc="Writing chunks"):
        start_id_a, stop_id_a = os.path.basename(chunk_path).split( "_")
        start_id_a = int(start_id_a)
        stop_id_a = int(stop_id_a)
        torch.save(gradients_subset[start_id_a:stop_id_a], chunk_path)
    
