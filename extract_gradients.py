import argparse
import os


from dotenv import load_dotenv
load_dotenv()

import wandb
from transformers import RobertaConfig,AutoConfig
from transformers import RobertaForMaskedLM
import torch
import traceback
import logging

from datasets import load_dataset
import util
import time
import util
import torch 

from util import get_checkpoints_hub
from util import DeterministicDataCollatorForLanguageModeling

from multiprocessing import Pool, Queue, Manager

import math

from transformers import RobertaTokenizerFast,GPT2TokenizerFast, DataCollatorForLanguageModeling,LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
 

os.environ["TOKENIZERS_PARALLELISM"] = "True"

parser = argparse.ArgumentParser("gradient_extraction")
parser.add_argument("model", help="A model on the hf hub. Format: username/name_curriculum")
parser.add_argument("dataset", help="A dataset on the hf hub. Format: username/name")
parser.add_argument("--dataset_split", help="The split to access", default="train")
parser.add_argument("checkpoint_nr", help="Id of the checkpoint to extract gradients for (starting at 0)",type=int)
parser.add_argument("--num_processes_gradients", help="Number of processes to use when obtaining gradients (one model per process)", type=int, nargs="?", const=1, default=12) # 12 w 4 gpus -> 3 models per gpu
parser.add_argument("--gradients_per_file", help="Number of gradients per output file", type=int, nargs="?", const=1, default=10000) # ~7.4 GB per file for BERT

args = parser.parse_args()


model_name = args.model.split("/")[-1]
dataset_name = args.dataset.split("/")[-1]
dataset_split_name = args.dataset_split

# create output dirs
gradient_output_dir = os.path.join("./gradients", model_name, dataset_name, dataset_split_name)
if not os.path.exists(gradient_output_dir):
    os.makedirs(gradient_output_dir)
# influence_output_dir = os.path.join("./influence", model_name, dataset_name, dataset_split_name) # this script skips computing existing results
# if not os.path.exists(influence_output_dir):
#     os.makedirs(influence_output_dir)



logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')




def get_loss_gradient(model, example,device):
    """Computes gradient of the loss function irt to the input embeddings for MLM.

    Args:
        model: A model with a `forward` method wich returns `loss`
        example: A string from the training data
        device: What GPU to use (e.g., cuda:0)

    Returns:
        A 1D tensor: gradient of the loss function irt to the input embeddings
    """
    
    model.zero_grad()
    
    input_ids, labels = data_collator((torch.tensor(example),)).values() # the data_collator used here applies the exact same mask used in the respective epoch
    inputs_embeds=model.get_input_embeddings().weight[input_ids].to(device)
    inputs_embeds.retain_grad()

    outputs = model.forward(
            inputs_embeds=inputs_embeds,
            labels=labels.to(device)
        )
    loss = outputs.loss
    loss.retain_grad()
    return torch.autograd.grad(loss, inputs_embeds, retain_graph=False)[0].squeeze()
    
   


def get_for_checkpoint(checkpoint_path, i_start, i_end, completion_times_gradients):
    """Calculates gradients at a given checkpoint for a given subset and stores it to disk

    Args:
        checkpoint_path: Path to the checkpoint folder
        i_start: Start id from the dataset
        i_end: Stop id from the dataset (non-inclusive)

    Raises:
        e: Any error but most likely OOM
    """                         
    
    if not any([a in args.model for a in ["llama", "OLMo"]]):
        data_collator.set_epoch(util.get_epoch(checkpoint_path)) # to ensure the same masking as during training
    try:
        gpu_id = gpu_queue.get()
        out_dir = os.path.join(gradient_output_dir, os.path.basename(checkpoint_path))
        out_path = os.path.join(gradient_output_dir, os.path.basename(checkpoint_path),str(i_start) + "_" + str(i_end))
        os.makedirs(out_dir,exist_ok=True)
        if os.path.isfile(out_path):
            gpu_queue.put(gpu_id)
            logging.info("Skipping {}, already generated".format(out_path) )
            return 


        device = "cuda:" + str(gpu_id)
        
        model = None
        if "llama" in args.model:
            model_config = AutoConfig.from_pretrained(checkpoint_path)
            model = LlamaForCausalLM(config=model_config).to(device)
        elif "OLMo" in args.model:
            print("loading model", flush=True)
            model = AutoModelForCausalLM.from_pretrained(args.model, revision=checkpoint_path, torch_dtype=torch.float16).to(device)
        else:
            model_config = AutoConfig.from_pretrained(checkpoint_path)
            model = RobertaForMaskedLM(config=model_config).to(device)
        model.train()

        start_time = time.time()
        gradients = torch.stack([get_loss_gradient(model, example,device).to(torch.bfloat16).detach().cpu() for example in dataset[i_start:i_end]["input_ids"]])#.cpu()
        print(f"Time to get gradients: {time.time() - start_time:.4f} s/chunk", flush=True)
        
        torch.save( gradients, out_path)
        gpu_queue.put(gpu_id)
        completion_times_gradients.append(time.time() - start_time)
   
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e





        
# def get_all_chunks(checkpoint_path):
#     """Returns output paths

#     Args:
#         checkpoint_path: A path to a model checkpoint

#     Returns:
#         A list of paths to the individual chunks generated
#     """
#     return [ os.path.join(gradient_output_dir, checkpoint_path.split("-")[-1],str(i) + "_" + str(i + args.gradients_per_file)) for i in range(0, len(dataset), args.gradients_per_file)]
    
tokenizer = None
data_collator = None
if "llama" in args.model:
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model, max_len=512)
    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

elif "OLMo" in args.model:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)
else:
    tokenizer = RobertaTokenizerFast.from_pretrained(args.model, max_len=512)
    data_collator = DeterministicDataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
) 


from util import tokenize_tulu_dataset

dataset = None

if "tulu" in args.dataset:
    dataset = tokenize_tulu_dataset(args.dataset, args.dataset_split)
else:
    dataset = load_dataset(args.dataset)[args.dataset_split] 
    dataset.set_transform(lambda x : tokenizer(x["text"], return_special_tokens_mask=True, truncation=True, padding="max_length", max_length=512))



checkpoints =  get_checkpoints_hub(args.model)



gpu_queue = Queue()



if __name__ == '__main__':
    
    assert args.num_processes_gradients//torch.cuda.device_count() > 0, "Need to assign at least as many processes as GPUs (change num_processes_gradients!)"
    # set up gpu queue to allocate gpus evenly between processes
    for _ in range(args.num_processes_gradients//torch.cuda.device_count()):

        for i in range(0,torch.cuda.device_count()):
            gpu_queue.put(i)
       

    manager = Manager()
    completion_times_gradients = manager.list() # the processes report back to the main process wich handles web logging

    pool_gradients = Pool(args.num_processes_gradients)
    
    checkpoint = checkpoints[args.checkpoint_nr]
    
    out_path = os.path.join(gradient_output_dir, os.path.basename(checkpoint))
   
    if os.path.isfile(out_path):
        logging.info("Skipping {}, already calculated".format(out_path) )
    else:
      
        run = wandb.init(project="babylm_gradient_extraction")
        run.name = os.path.join(args.model, os.getenv("SLURM_JOB_NAME", "?"))
        logging.info("Getting gradients for checkpoint-{}".format(checkpoint))

        # create tasks for subprocesses
        tasks_gradients = [(checkpoint,i, i + args.gradients_per_file, completion_times_gradients) for i in range(0, len(dataset), args.gradients_per_file)]
        r = pool_gradients.starmap_async(get_for_checkpoint, tasks_gradients)   

        # web logging
        while not r.ready(): # loop to not block the main process
            while len(completion_times_gradients) > 0:
                run.log({"gradients/time_per_chunk": completion_times_gradients.pop()},commit=True)
            run.log({"gradients/pool_ram_usage":util.get_pool_memory_usage(pool_gradients)}, commit=True)
            time.sleep(10)  
        
        logging.info("Got gradients for checkpoint-{}".format(checkpoint))
    pool_gradients.close()
    pool_gradients.join()