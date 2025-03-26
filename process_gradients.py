import argparse
import os

import setproctitle
from dotenv import load_dotenv
load_dotenv()
import wandb

from util import batch 

from functools import partial

import time 
import datetime
import os
from pathlib import Path
import torch
from itertools import cycle
from util import get_checkpoints_hub
import sys
from util import DeterministicDataCollatorForLanguageModeling
import sys
import shutil
import json
from datasets import load_dataset

import torch
import util



import shutil
import torch 
import time
from concurrent.futures import ThreadPoolExecutor
import random

import logging
import util


import time

import datasets

parser = argparse.ArgumentParser("gradient_extraction")

parser.add_argument("model", help="A model on the hf hub. Format: username/name_curriculum")
parser.add_argument("dataset_train", help="A dataset on the hf hub. Format: username/name")
parser.add_argument("--dataset_train_split", help="The split to access", default="train")
parser.add_argument("checkpoint_nr", help="Id of the checkpoint to extract gradients for (starting at 0)",type=int)
parser.add_argument("--num_processes", help="Number of processes to use when doing dot product (runs on cpu)", type=int, nargs="?", const=1, default=1)
parser.add_argument("--gradients_per_file", help="Number of gradients per output file", type=int, nargs="?", const=1, default=1000)
parser.add_argument("--batch_size", help="How many chunks each subprocess will keep in memory", type=int, nargs="?", const=1, default=200)

parser.add_argument("--mode", help="One of 'single', 'mean', 'mean_normalized'.If 'mean', mean influence of individual train on all examples in test; if 'single' 1 train -> 1 test", default="single")
parser.add_argument("--dataset_test", help="A dataset on the hf hub. If supplied, returns one score per test instance. Format: username/name", default=None)
parser.add_argument("--dataset_test_split", help="The split to access", default="test")

parser.add_argument("--delete_test_on_success", default=False, action='store_true')
parser.add_argument("--delete_train_on_success", default=False, action='store_true')

parser.add_argument("--test", help="The split to access", default=False)
parser.add_argument("--test_dataset_size", help="The split to access", default=0, type=int)

parser.add_argument("--random_projection", default=False, action='store_true')
parser.add_argument("--proj_dim", type=int, nargs="?", const=1, default=2**14)

args = parser.parse_args()
print("args", args, flush=True)


model_name = args.model.split("/")[-1]
dataset_train_name = args.dataset_train.split("/")[-1]
dataset_train_split_name = args.dataset_train_split
dataset_test_name = args.dataset_test.split("/")[-1]
dataset_test_split_name = args.dataset_test_split


gradient_output_dir_train = os.path.join("./gradients", str(args.proj_dim) if args.random_projection else "full", model_name, dataset_train_name, dataset_train_split_name)
gradient_output_dir_test = os.path.join("./gradients", str(args.proj_dim) if args.random_projection else "full", model_name, dataset_test_name, dataset_test_split_name)

influence_dir = "./influence" if args.mode == "single" else "./influence_" + args.mode 

if not os.path.exists(influence_dir):
    os.makedirs(influence_dir)
    
influence_output_dir = os.path.join(influence_dir, model_name, "_".join([dataset_train_name, dataset_train_split_name, dataset_test_name, dataset_test_split_name]))
if not os.path.exists(influence_output_dir):
    os.makedirs(influence_output_dir)

logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')



# TODO we only need to know the lenght of the dataset here

dataset_train = None
if args.test:
    dataset_train = torch.zeros(args.test_dataset_size)
else:

    dataset_train = load_dataset(args.dataset_train, split=args.dataset_train_split)
        
dataset_test = None

if args.dataset_test is not None:
    dataset_test = load_dataset(args.dataset_test, split=args.dataset_test_split)
else:
    dataset_test = dataset_train









def calc_partial(tasks, subtasks):
    """Calculates mean TracinCP influence for the items in each chunk in `tasks` on all other items in the training data. Effectively just does items.dot(items.T).mean() but in batches.

    Args:
        tasks: Paths to chunks to be kept in memory at all times
        subtasks: Paths to all chunks in a checkpoint to be streamed 

    Raises:
        e: Any error

    Returns:
        A tuple of tasks (same as input) and (partial) results 
    """


    device = "cpu"
    with torch.no_grad():
        start_time = time.time()
      
        # load task (of batch_size chunks)
        def load_fn(chunk_path): 
            cache_path = os.path.join("/tmp/gradients/",chunk_path)
            while True:
                try:
                    return torch.load(cache_path,weights_only=True).flatten(1)
                except:
                   
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                  
                    if not os.path.isfile(cache_path):
                        shutil.copyfile(chunk_path, cache_path)

        logging.debug(f"Loading traning tasks...")
        chunks_a = []
        with ThreadPoolExecutor(max_workers=32) as executor:
            chunks_a = list(executor.map(lambda task: (load_fn(task[0]), task[1], task[2]), tasks))
        
        logging.debug(f"...time to load task: {time.time() - start_time:.4f} seconds")
  
 
        if args.mode == "single":
            results = []
            for chunk_path_b,start_id_b, stop_id_b in subtasks:
                chunk_b = load_fn(chunk_path_b) # reuse the chunks in chunks_a if they are in this subtask
                for chunk_a, start_id_a, stop_id_a in chunks_a:
                    results.append((torch.matmul(chunk_a, chunk_b.T), start_id_a, stop_id_a, start_id_b, stop_id_b))
            return results
        elif args.mode == "mean":
            assert len(subtasks) == 1        
            chunk_b = torch.load(subtasks[0][0],weights_only=False).flatten(0)
            results = [torch.mv(chunk_a, chunk_b) for i, (chunk_a, _,_) in enumerate(chunks_a)]
            return (tasks, results)
        elif args.mode == "mean_normalized":
            assert len(subtasks) == 1
            chunk_b = torch.load(subtasks[0][0],weights_only=False).flatten(0) # b stored pre-normalized
            results = [torch.mv(torch.nn.functional.normalize(chunk_a), chunk_b) for i, (chunk_a, _,_) in enumerate(chunks_a)] # b not      
            return (tasks, results)
        else:
            raise NotImplementedError








if __name__ == '__main__':   
    checkpoints =  get_checkpoints_hub(args.model) if not args.test else ["test/test/test/test"]
    checkpoint = checkpoints[args.checkpoint_nr]
        

    out_path = os.path.join(influence_output_dir, os.path.basename(checkpoint))
    if os.path.isfile(out_path):
        logging.info("Skipping {}, already calculated".format(out_path) )
        
        
    else:
        run = wandb.init(project="babylm_influence_computation")
        run.name = os.path.join(args.model, os.getenv("SLURM_JOB_NAME", "?"))

        logging.info("Calculating influence for checkpoint-{}".format(checkpoint))

        
        # create batches (over training dataset)
        tasks = []
        chunks_train = [ os.path.join(gradient_output_dir_train, os.path.basename(checkpoint), str(i) + "_" + str(i + args.gradients_per_file)) for i in range(0, len(dataset_train), args.gradients_per_file)]

        for chunk_path_a in chunks_train:
            start_id_a, stop_id_a = os.path.basename(chunk_path_a).split( "_")
            start_id_a = int(start_id_a)
            stop_id_a = int(stop_id_a)
            tasks.append((chunk_path_a,start_id_a, stop_id_a, ))

        subtasks = []
        if args.mode == "single":
            chunks_test = [ os.path.join(gradient_output_dir_test, os.path.basename(checkpoint), str(i) + "_" + str(i + args.gradients_per_file)) for i in range(0, len(dataset_test), args.gradients_per_file)]
            for chunk_path_b in chunks_test:
                start_id_b, stop_id_b = os.path.basename(chunk_path_b).split( "_")
                start_id_b = int(start_id_b)
                stop_id_b = int(stop_id_b)
                subtasks.append((chunk_path_b,start_id_b, stop_id_b, ))
        elif "mean" in args.mode:
            subtasks = [(os.path.join(gradient_output_dir_test, os.path.basename(checkpoint), args.mode),0,1)]
        else:
            raise NotImplementedError

        r = []
        for tasks in batch(tasks, args.batch_size):
            start_time = time.time()
            r.append(calc_partial(tasks, subtasks))
            run.log({"influence/time_per_chunk": time.time()-start_time},commit=False)
            run.log({"influence/time_per_example": (time.time()-start_time)/args.gradients_per_file},commit=True)



        result_checkpoint = None
        if args.mode == "single":
            result_checkpoint  = torch.empty((len(dataset_train), len(dataset_test)))
        else:
            result_checkpoint = torch.zeros((len(dataset_train)))
        if args.mode == "single":
            for result, start_id_a, stop_id_a, start_id_b, stop_id_b in results[0]:
                result_checkpoint[start_id_a:start_id_a+result.shape[0], start_id_b:start_id_b+result.shape[1]] = result
            result_checkpoint = (result_checkpoint / len(dataset_test)).unsqueeze(0) 
        else:
            for rr in results:
                for task, result in zip(*rr):
                    chunk_path_a, start_id_a, stop_id_a = task
                    print("chunk_path_a, start_id_a, stop_id_a",chunk_path_a, start_id_a, stop_id_a, flush=True)
                    result_checkpoint[start_id_a:(start_id_a + result.shape[0])] += result #  the stop_ids are taken from the task description in if.ipynb and can therefore be higher than the actual lenght
            result_checkpoint = result_checkpoint.unsqueeze(0) 
            
    
          
        torch.save(result_checkpoint, out_path)
        logging.info("Saved influence for checkpoint".format(out_path))
       
        if args.delete_train_on_success:
            shutil.rmtree(gradient_output_dir_train, ignore_errors=True)
        if args.delete_test_on_success:
            shutil.rmtree(gradient_output_dir_test, ignore_errors=True)
        

        run.finish()

