import argparse
import os
import setproctitle
from dotenv import load_dotenv
load_dotenv()



parser = argparse.ArgumentParser("gradient_extraction")

parser.add_argument("model", help="A model on the hf hub. Format: username/name_curriculum")
parser.add_argument("dataset", help="A dataset on the hf hub. Format: username/name")
parser.add_argument("--num_processes_gradients", help="Number of processes to use when obtaining gradients (one model per process)", type=int, nargs="?", const=1, default=6)
parser.add_argument("--num_processes_merge", help="Number of processes to use when doing dot product (runs on cpu)", type=int, nargs="?", const=1, default=4)
# parser.add_argument("--cuda_visible_devices", help="Comma seperated GPU ids to use", nargs="?", const=1, default="0,1")
parser.add_argument("--gradients_per_file", help="Number of gradients per output file", type=int, nargs="?", const=1, default=10000)
parser.add_argument("--batch_size", help="How many chunks each subprocess will keep in memory", type=int, nargs="?", const=1, default=20)

args = parser.parse_args()

import json
import wandb
from datasets import load_dataset
import datasets
model_name = args.dataset.split("/")[-1]
# create output dirs


if not os.path.exists("./gradients"):
    os.makedirs("./gradients")
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


def get_loss_gradient(model, example,device):
    """Computes gradient of the loss function irt to the input

    Args:
        model: A model with a `forward` method wich returns `loss`
        example: A string from the training data
        device: What GPU to use

    Returns:
        A 1D tensor: gradient of the loss function irt to the input
    """
    
    model.zero_grad()
    
    input_ids, labels = data_collator((torch.tensor(example),)).values()
    inputs_embeds=model.get_input_embeddings().weight[input_ids].to(device)
    inputs_embeds.retain_grad()

    outputs = model.forward(
            inputs_embeds=inputs_embeds,
            labels=labels.to(device)
        )
    loss = outputs.loss
    loss.retain_grad()
    return  torch.autograd.grad(loss, inputs_embeds, retain_graph=False)[0].squeeze()

import datasets
dataset = load_dataset(args.dataset)["train"]
dataset.set_transform(lambda x : tokenizer(x["text"], return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=512))
import util
def get_for_checkpoint(checkpoint_path, i_start, i_end, completion_times_gradients):
    """Calculates gradients at a given checkpoint for a given subset and stores it to disk

    Args:
        checkpoint_path: Path to the checkpoint folder
        i_start: Start id from the dataset
        i_end: Stop id from the dataset (non-inclusive)

    Raises:
        e: Any error but most likely OOM
    """
    setproctitle.setproctitle("(loris:0,1) max 12516 MiB/GPU can share")
                         


    data_collator.set_epoch(util.get_epoch(checkpoint_path)) # to ensure the same masking as during training
    try:
        gpu_id = queue.get()
        out_dir = os.path.join(gradient_output_dir, checkpoint_path.split("-")[-1])
        out_path = os.path.join(gradient_output_dir, checkpoint_path.split("-")[-1],str(i_start) + "_" + str(i_end))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if os.path.isfile(out_path):
            queue.put(gpu_id)
            logging.info("Skipping {}, already generated".format(out_path) )
            return 


        device = "cuda:" + str(gpu_id)
        model_config = AutoConfig.from_pretrained(checkpoint_path)
        model = RobertaForMaskedLM(config=model_config).to(device)
        model.train()

        start_time = time.time()
        gradients = torch.stack([get_loss_gradient(model, example,device).to(torch.bfloat16).detach().cpu() for example in dataset[i_start:i_end]["input_ids"]])#.cpu()
        print(f"Time to get gradients: {time.time() - start_time:.4f} s/chunk", flush=True)
        
        torch.save( gradients, out_path)
        queue.put(gpu_id)
        completion_times_gradients.append(time.time() - start_time)
   
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e


##########################################


import torch 
import time

def calc_partial(tasks, subtasks,completion_times_influence):
    """Calculates mean TracinCP influence for the items in each chunk in `tasks` on all other items in the training data. Effectively just does items.dot(items.T).mean() but in batches.

    Args:
        tasks: Paths to chunks to be kept in memory at all times
        subtasks: Paths to all chunks in a checkpoint to be streamed 

    Raises:
        e: Any error

    Returns:
        A tuple of tasks (same as input) and (partial) results 
    """


    setproctitle.setproctitle("(loris) max 650 GB RAM total")
    device = "cpu"
    
    try:
        with torch.no_grad():
            start_time = time.time()
            try:
                chunks_a = [torch.load(chunk_path_a, weights_only=True,map_location=device).flatten(1) for (chunk_path_a,start_id_a, stop_id_a) in tasks]
            except Exception as e:
                logging.error("{} seems corrupted (recompute!)".format(tasks) ) # this happens if the gradient extraction script was killed during torch.save
                logging.error(traceback.format_exc())
                
                raise e
            logging.info(f"Time to load task: {time.time() - start_time:.4f} seconds")
            results = [torch.zeros((chunk_a.shape[0])).to(device) for chunk_a in chunks_a]
            start_time = time.time()
            for chunk_path_b,start_id_b, stop_id_b in subtasks:
                try:
                    chunk_b = torch.load(chunk_path_b, weights_only=True,map_location=device).flatten(1)
                except Exception as e:
                    logging.error("{} seems corrupted (recompute!)".format(chunk_path_b) )
                    logging.error(traceback.format_exc())
                    # this happens if the gradient extraction script was killed during torch.save
                    
                    raise e
                for i, chunk_a in enumerate(chunks_a):
                    results[i]  += torch.einsum('ik, kj -> i', chunk_a, chunk_b.T)
            logging.info(f"Time to einsum: {time.time() - start_time:.4f} seconds; {(time.time() - start_time)/len(subtasks):.4f} s/chunk")
            completion_times_influence.append(time.time() - start_time)
            return (tasks, results)

    except Exception as e:
        logging.error(traceback.format_exc())
        raise e


import math
from sklearn.utils import gen_even_slices
def batch(lst, batch_size):
        for i in gen_even_slices(len(lst), math.ceil(len(lst)/batch_size)):
            yield lst[i]
        
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




from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained(args.model, max_len=512)


from util import get_epoch_checkpoints_hub
checkpoints =  get_epoch_checkpoints_hub(args.model)

from util import DeterministicDataCollatorForLanguageModeling

data_collator = DeterministicDataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)



import itertools
import subprocess
from util import get_epoch_checkpoints
queue = Queue()

for _ in range(args.num_processes_gradients//torch.cuda.device_count()):
    for i in range(torch.cuda.device_count()):
        queue.put(i)

from multiprocessing import Pool, Manager


if __name__ == '__main__':
    run = wandb.init(project="babylm_influence_computation")
    run.name = args.model
    manager = Manager()
    completion_times_gradients = manager.list()
    completion_times_influence = manager.list() 

    pool_gradients = Pool(args.num_processes_gradients)
    pool_merge = Pool(args.num_processes_merge)
    for checkpoint_nr, checkpoint in enumerate(checkpoints):
        run.log({"gradients/checkpoint": checkpoint_nr},commit=True)
        #### 1. extract gradients ###
        out_path = os.path.join(influence_output_dir, checkpoint.split("-")[-1])
        if os.path.isfile(out_path):
            logging.info("Skipping {}, already calculated".format(out_path) )
            continue
        logging.info("Getting gradients for checkpoint-{}".format(checkpoint))
        # specify tasks for subprocesses
        tasks_gradients = [(checkpoint,i, i + args.gradients_per_file, completion_times_gradients) for i in range(0, len(dataset), args.gradients_per_file)]
        
    
        r = pool_gradients.starmap_async(get_for_checkpoint, tasks_gradients)   
        while not r.ready(): # to enable logging troughput: loop to not block the main process
            #print("Tasks still running...", completion_times_gradients)
            while len(completion_times_gradients) > 0:
                run.log({"gradients/time_per_chunk": completion_times_gradients.pop()},commit=True)
            time.sleep(10)   
        run.log({"influence/checkpoint": checkpoint_nr},commit=True)




         #### 2. calculate mean influence ###
        logging.info("Calculating influence for checkpoint-{}".format(checkpoint))
    
        
        # specify tasks for subprocesses
        jobs = []
        subtasks = []
        for chunk_path_b in get_all_chunks(checkpoint):
            #print("chunk_path_b",chunk_path_b)
            start_id_b, stop_id_b = os.path.basename(chunk_path_b).split( "_")
            start_id_b = int(start_id_b)
            stop_id_b = int(stop_id_b)
            subtasks.append((chunk_path_b,start_id_b, stop_id_b, ))
    
        for tasks in batch(subtasks, args.batch_size):
            jobs.append((tasks, subtasks,completion_times_influence))

        def onPoolDone(r, out_path,checkpoint):
            print("onPoolDone",r, flush=True)
            result_checkpoint = torch.zeros((len(dataset)))
            for rr in r:
                for task, result in zip(*rr):
                    chunk_path_a, start_id_a, stop_id_a = task
                    result_checkpoint[start_id_a:(start_id_a + result.shape[0])] += result #  the stop_ids are taken from the task description in if.ipynb and can therefore be higher than the actual lenght
            result_checkpoint = (result_checkpoint / len(dataset)).unsqueeze(0)   
            torch.save(result_checkpoint, out_path)
            logging.info("Saved influence for checkpoint".format(out_path))
            logging.info("Deleting gradients for {}".format(checkpoint))
            
            for filename in os.listdir(os.path.join(gradient_output_dir, checkpoint.split("-")[-1])):
                file_path = os.path.join(gradient_output_dir, checkpoint.split("-")[-1], filename)
                os.remove(file_path)
        
        r = pool_merge.starmap_async(calc_partial, jobs, callback=lambda result,out_path=out_path,checkpoint=checkpoint: onPoolDone(result, out_path,checkpoint))
        while not r.ready(): # to enable logging troughput: loop to not block the main process
            #print("Tasks still running...", completion_times_influence)
            while len(completion_times_influence) > 0:
                run.log({"influence/time_per_batch": completion_times_influence.pop()},commit=True)
            time.sleep(10)   
        

    
    run.finish()
