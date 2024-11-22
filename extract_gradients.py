import argparse
import os
import setproctitle
from dotenv import load_dotenv
load_dotenv()



parser = argparse.ArgumentParser("gradient_extraction")

parser.add_argument("model", help="A model on the hf hub. Format: username/name_curriculum")
parser.add_argument("dataset", help="A dataset on the hf hub. Format: username/name")
parser.add_argument("checkpoint_nr", help="Id of the checkpoint to extract gradients for (starting at 0)",type=int)
parser.add_argument("--num_processes_gradients", help="Number of processes to use when obtaining gradients (one model per process)", type=int, nargs="?", const=1, default=6)
# parser.add_argument("--cuda_visible_devices", help="Comma seperated GPU ids to use", nargs="?", const=1, default="0,1")
parser.add_argument("--gradients_per_file", help="Number of gradients per output file", type=int, nargs="?", const=1, default=10000)

args = parser.parse_args()

import json

from datasets import load_dataset
import datasets
model_name = args.dataset.split("/")[-1]
# create output dirs



gradient_output_dir = os.path.join("./gradients", model_name)
if not os.path.exists(gradient_output_dir):
    os.makedirs(gradient_output_dir)
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
        os.makedirs(out_dir,exist_ok=True)
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


from util import get_checkpoints_hub
checkpoints =  get_checkpoints_hub(args.model)

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

import sys
if __name__ == '__main__':
    manager = Manager()
    completion_times_gradients = manager.list()

    pool_gradients = Pool(args.num_processes_gradients)

    
    checkpoint = checkpoints[args.checkpoint_nr]
    

    out_path = os.path.join(influence_output_dir, checkpoint.split("-")[-1])
    if os.path.isfile(out_path):
        logging.info("Skipping {}, already calculated".format(out_path) )
    else:
        logging.info("Getting gradients for checkpoint-{}".format(checkpoint))
        # specify tasks for subprocesses
        tasks_gradients = [(checkpoint,i, i + args.gradients_per_file, completion_times_gradients) for i in range(0, len(dataset), args.gradients_per_file)]
        

        r = pool_gradients.starmap_async(get_for_checkpoint, tasks_gradients)   
        while not r.ready(): # to enable logging troughput: loop to not block the main process
            #print("Tasks still running...", completion_times_gradients)
            while len(completion_times_gradients) > 0:
                logging.info({"gradients/time_per_chunk": completion_times_gradients.pop()})
           
        logging.info("Got gradients for checkpoint-{}".format(checkpoint))