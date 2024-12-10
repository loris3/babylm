
from multiprocessing import Pool, Queue, Manager
import argparse
import torch
import os
from util import batch 

from dotenv import load_dotenv
load_dotenv()

import time
from datasets import load_dataset
import traceback

parser = argparse.ArgumentParser("difficulty")
parser.add_argument("--num_processes", help="Number of processes to use when doing dot product (runs on cpu)", type=int, nargs="?", const=1, default=4)
parser.add_argument("--cuda_visible_devices", help="Comma seperated GPU ids to use", nargs="?", const=1, default="0,1")
parser.add_argument("model", help="A model on the hf hub. Format: username/name_curriculum")
parser.add_argument("dataset", help="A dataset on the hf hub. Format: username/name")
parser.add_argument("--batch_size", help="How many documents to calculate metrics for at a time per process", type=int, nargs="?", const=1, default=1000000)
parser.add_argument("--batch_size_gpu", help="Batch size for GPU", type=int, nargs="?", const=1, default=2**18)
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

gpu_queue = Queue()

import logging
logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')



from evaluate import load
import pandas as pd
def calc_perplexity(task):
    try:
        gpu_id = gpu_queue.get()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        index, input_texts = zip(*[(i,s) for i,s in task  if s!='']) # Each input text must be at least one token long
        
        
        perplexity = load("perplexity", module_type="metric", batch_size=args.batch_size_gpu)
        df = pd.DataFrame(
            perplexity.compute(predictions=input_texts, model_id=args.model,max_length=512)["perplexities"],
            index = index
            )
        df.columns=["perplexity"]
        df.index.name = "document_id"

        

        gpu_queue.put(gpu_id)
        return df
    except:
        print(traceback.format_exc(), flush=True)

if __name__ == '__main__':
    

    # set up gpu queue to allocate gpus evenly between processes
    for _ in range(args.num_processes//torch.cuda.device_count()):
        for i in [int(j) for j in os.getenv("CUDA_VISIBLE_DEVICES").split(",")]:
            gpu_queue.put(i)

    manager = Manager()
   
    pool = Pool(args.num_processes)
    

    
    out_path = os.path.join("./difficulty", os.path.basename(args.dataset), os.path.basename(args.model))
    print(out_path)
    if os.path.isdir(out_path):
        logging.info("Skipping {}, already calculated".format(out_path) )
    else:
    
        # create tasks for subprocesses
        dataset = load_dataset(args.dataset)["train"]
        tasks = batch(list(enumerate(dataset["text"])), args.batch_size)
        aa = list(pool.map(calc_perplexity, tasks))
        print("aa", flush=True)
        df_perplexity = pd.concat(aa)
        
        os.makedirs(out_path)
        logging.info("Saving metrics for {}".format(args.dataset))
        df_perplexity.to_parquet(os.path.join(out_path,"perplexity"))
        logging.info("Got metrics for {}".format(args.dataset))
    pool.close()
    pool.join()