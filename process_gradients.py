import argparse
import os

import setproctitle
from dotenv import load_dotenv
load_dotenv()
import wandb

from util import batch 

from functools import partial
from multiprocessing.pool import ThreadPool


parser = argparse.ArgumentParser("gradient_extraction")

parser.add_argument("model", help="A model on the hf hub. Format: username/name_curriculum")
parser.add_argument("dataset", help="A dataset on the hf hub. Format: username/name")
parser.add_argument("checkpoint_nr", help="Id of the checkpoint to extract gradients for (starting at 0)",type=int)
parser.add_argument("--num_processes", help="Number of processes to use when doing dot product (runs on cpu)", type=int, nargs="?", const=1, default=1)
# parser.add_argument("--cuda_visible_devices", help="Comma seperated GPU ids to use", nargs="?", const=1, default="0,1")
parser.add_argument("--gradients_per_file", help="Number of gradients per output file", type=int, nargs="?", const=1, default=10000)
parser.add_argument("--batch_size", help="How many chunks each subprocess will keep in memory", type=int, nargs="?", const=1, default=108)

args = parser.parse_args()

import json

from datasets import load_dataset
import datasets
model_name = args.dataset.split("/")[-1]
# create output dirs



gradient_output_dir = os.path.join("./gradients", model_name)
if not os.path.exists(gradient_output_dir):
    os.makedirs(gradient_output_dir)

if not os.path.exists("./influence"):
    os.makedirs("./influence")
influence_output_dir = os.path.join("./influence", model_name)
if not os.path.exists(influence_output_dir):
    os.makedirs(influence_output_dir)


# os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
os.environ["TOKENIZERS_PARALLELISM"] = "True"



# run.save()


from transformers import RobertaConfig,AutoConfig
from transformers import RobertaForMaskedLM
import torch
import traceback

import logging
logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')




import util

from tqdm import tqdm
import time

import datasets
dataset = load_dataset(args.dataset)["train"]
dataset.set_transform(lambda x : tokenizer(x["text"], return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=512))
import util


##########################################


import torch 
import time
from concurrent.futures import ThreadPoolExecutor

def calc_partial(tasks, subtasks,completion_times_influence, einsum_times_influence):
    """Calculates mean TracinCP influence for the items in each chunk in `tasks` on all other items in the training data. Effectively just does items.dot(items.T).mean() but in batches.

    Args:
        tasks: Paths to chunks to be kept in memory at all times
        subtasks: Paths to all chunks in a checkpoint to be streamed 

    Raises:
        e: Any error

    Returns:
        A tuple of tasks (same as input) and (partial) results 
    """


    setproctitle.setproctitle("(loris) max 500 GB RAM total")
    device = "cpu"
    

    with torch.no_grad():
        chunks_a = None
    
        start_time = time.time()
        # load task (of batch_size chunks)
        load_fn = lambda chunk_path: torch.load(chunk_path, weights_only=True, map_location=device).flatten(1)

        with ThreadPoolExecutor(max_workers=50) as executor:
            chunks_a = list(executor.map(lambda task: load_fn(task[0]), tasks))

        print("num chunks", len(chunks_a), flush=True)
        total_size_gb = lambda tensors: sum(t.element_size() * t.numel() for t in tensors) / (1024**3)
        print("size", total_size_gb(chunks_a), flush=True)
        # chunks_a = [torch.load(chunk_path_a, weights_only=True,map_location=device).flatten(1) for (chunk_path_a,start_id_a, stop_id_a) in tasks]

        logging.info(f"Time to load task: {time.time() - start_time:.4f} seconds")
        print(f"Time to load task: {time.time() - start_time:.4f} seconds", flush=True)
        results = [torch.zeros((chunk_a.shape[0])).to(device) for chunk_a in chunks_a]
        start_time = time.time()

        tasks_paths = list(zip(*tasks))[0]

        # reuse the chunks in chunks_a if they are in this subtask
        def load_cached(chunk_path):
            if chunk_path in tasks_paths:
                return chunks_a[tasks_paths.index(chunk_path)]
            else:
                return torch.load(chunk_path, weights_only=True, map_location=device).flatten(1)

        for chunk_path_b,start_id_b, stop_id_b in subtasks:
            chunk_b = load_cached(chunk_path_b)
            start_time_einsum = time.time()
            for i, chunk_a in enumerate(chunks_a):
                results[i]  += torch.einsum('ik, kj -> i', chunk_a, chunk_b.T)
            einsum_times_influence.append(time.time() - start_time_einsum)
        logging.info(f"Time to einsum: {time.time() - start_time:.4f} seconds; {(time.time() - start_time)/len(subtasks):.4f} s/chunk")
        completion_times_influence.append(time.time() - start_time)
        return (tasks, results)






def get_all_chunks(checkpoint_path):
    return [ os.path.join(gradient_output_dir, checkpoint_path.split("-")[-1],str(i) + "_" + str(i + args.gradients_per_file)) for i in range(0, len(dataset), args.gradients_per_file)]
    
    return [str(x) for x in Path(checkpoint_path).glob("*")]





#########################################
from multiprocessing import Pool, current_process, Queue
import time 
import datetime
import os
from pathlib import Path
import torch
from itertools import cycle


import sys

from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(args.model, max_len=512)


from util import get_checkpoints_hub
checkpoints =  get_checkpoints_hub(args.model)

from util import DeterministicDataCollatorForLanguageModeling

data_collator = DeterministicDataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)



import itertools
import subprocess
import sys

from multiprocessing import Pool, Manager
import glob

import shutil
if __name__ == '__main__':
    
    manager = Manager()

    completion_times_influence = manager.list() 
    einsum_times_influence = manager.list() 


    pool_merge = Pool(args.num_processes)

    print(sum(os.path.isdir(os.path.join(gradient_output_dir, item)) for item in os.listdir(gradient_output_dir)), "gradient folders waiting", flush=True)
    checkpoint = checkpoints[args.checkpoint_nr]
        
    # copy gradients to local filesystem

    gradient_output_dir_local = os.path.join("/tmp/gradients", model_name)
    path_remote = os.path.join(gradient_output_dir, checkpoint.split("-")[-1])
    path_local = os.path.join(gradient_output_dir_local, checkpoint.split("-")[-1])

    ##########################
    # if not os.path.exists(path_local):
    #     os.makedirs(path_local)


    # def copy_tp(src, dst=path_local):
    #     dest_file = os.path.join(dst, os.path.basename(src))
    #     if not os.path.exists(dest_file):
    #         shutil.copy(src, dest_file)
    
    # files = glob.glob(os.path.join(path_remote, '*'))
    # print("copy", path_remote, path_local,files)
    # with ThreadPool(20) as p:
    #     p.map(copy_tp, files)
    # print("copied", flush=True)

    # gradient_output_dir = gradient_output_dir_local

    ###############
    print(sum(os.path.isdir(os.path.join(gradient_output_dir, item)) for item in os.listdir(gradient_output_dir)), "gradient folders waiting", flush=True)
    out_path = os.path.join(influence_output_dir, checkpoint.split("-")[-1])
    if os.path.isfile(out_path):
        logging.info("Skipping {}, already calculated".format(out_path) )
        
        
    else:
        run = wandb.init(project="babylm_influence_computation")
        run.name = os.path.join(args.model, os.getenv("SLURM_JOB_NAME", "?"))

            #### 2. calculate mean influence ###
        logging.info("Calculating influence for checkpoint-{}".format(checkpoint))

        
        # specify tasks for subprocesses
        jobs = []
        subtasks = []
        print("get_all_chunks(checkpoint)", len(get_all_chunks(checkpoint)), flush=True)
        for chunk_path_b in get_all_chunks(checkpoint):
            #print("chunk_path_b",chunk_path_b)
            start_id_b, stop_id_b = os.path.basename(chunk_path_b).split( "_")
            start_id_b = int(start_id_b)
            stop_id_b = int(stop_id_b)
            subtasks.append((chunk_path_b,start_id_b, stop_id_b, ))

        print("subtasks", len(subtasks), flush=True)
        for tasks in batch(subtasks, args.batch_size):
            print("tasks", len(tasks),flush=True)
            jobs.append((tasks, subtasks,completion_times_influence, einsum_times_influence))
       
        print("jobs", len(jobs),flush=True)



        r = pool_merge.starmap_async(calc_partial, jobs)
        
        while not r.ready(): # to enable logging troughput: loop to not block the main process
            #print("Tasks still running...", completion_times_influence)
            while len(completion_times_influence) > 0:
                run.log({"influence/time_per_batch": completion_times_influence.pop()}, commit=True)
            while len(einsum_times_influence) > 0:
                run.log({"influence/time_einsum_per_chunk": einsum_times_influence.pop()}, commit=True)
            run.log({"influence/pool_ram_usage":util.get_pool_memory_usage(pool_merge)}, commit=True)
            time.sleep(10)   
        print("onPoolDone", flush=True)
        result_checkpoint = torch.zeros((len(dataset)))
        for rr in r.get():
            for task, result in zip(*rr):
                chunk_path_a, start_id_a, stop_id_a = task
                result_checkpoint[start_id_a:(start_id_a + result.shape[0])] += result #  the stop_ids are taken from the task description in if.ipynb and can therefore be higher than the actual lenght
        result_checkpoint = (result_checkpoint / len(dataset)).unsqueeze(0)   
        torch.save(result_checkpoint, out_path)
        logging.info("Saved influence for checkpoint".format(out_path))
        logging.info("Deleting gradients for {}".format(checkpoint))
        
        shutil.rmtree(path_remote, ignore_errors=True)
        
    pool_merge.close()
    pool_merge.join()
    logging.info("Got influence for checkpoint-{}".format(checkpoint))
    



