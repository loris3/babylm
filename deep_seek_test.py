import argparse
import os


from dotenv import load_dotenv
load_dotenv()

import wandb

import torch
import traceback
import logging

from datasets import load_dataset
import util
import time
import util
import torch 


from multiprocessing import Pool, Queue, Manager

import math


from transformers import AutoModelForCausalLM, AutoTokenizer,DataCollatorForLanguageModeling
 


logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')




if __name__ == '__main__':

    model_name = "deepseek-ai/DeepSeek-V3"

    tokenizer = None
    data_collator = None

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)#.to(device)
    prompt = "A computer would deserve to be "
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_length=200)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    print(generated_text, flush=True)

