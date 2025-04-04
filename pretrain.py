import os
print(os.environ)
from dotenv import load_dotenv
load_dotenv()

import os
print(os.environ)
import argparse
import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from torch.utils.data import Dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import torch
from random import randrange
import cloudpickle
from datasets import load_dataset
import random
from transformers import RobertaForMaskedLM, DataCollatorForLanguageModeling, LlamaForCausalLM, LlamaTokenizerFast
import evaluate

from collections import defaultdict
import math

import util
from torch.utils.data import DataLoader
from transformers.trainer_utils import (seed_worker)
from tokenizers import ByteLevelBPETokenizer
import datasets
from datasets import load_dataset
from transformers import RobertaTokenizerFast
from transformers import Trainer, TrainingArguments
from torch.utils.data import SequentialSampler
from transformers import GPT2TokenizerFast
import json
from transformers import RobertaConfig, LlamaConfig
from tokenizers import (Tokenizer, decoders, models, pre_tokenizers,
                        processors, trainers)
from tokenizers.normalizers import NFKC

parser = argparse.ArgumentParser("pretraining")
parser.add_argument("dataset", help="A dataset on the hf hub. Format: username/name")
parser.add_argument("curriculum", help="The curriculum to use. Filename in the dataset repo. Examples: curriculum.pt or random.pt")
parser.add_argument("--model_type", help="One of 'roberta', 'llama'", default="roberta")
args = parser.parse_args()


os.environ["WANDB_PROJECT"]="pretraining"

datasets = load_dataset(args.dataset)

model_path =  os.path.join("models/", args.dataset.split("/")[-1] + ("_" + args.model_type) + "_" +args.curriculum.split(".")[0])
if not os.path.exists(model_path):
    os.makedirs(model_path)




# Load or create tokenizer
tokenizer = None
try:
    if args.model_type == "llama":
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path, max_len=512)
        tokenizer.model_max_length = 128
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = "<pad>"
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(model_path, max_len=512)
except:
    def batch_iterator(batch_size=1000):
            for i in range(0, len(dataset_tokenizer), batch_size):
                yield dataset_tokenizer[i: i + batch_size]["text"]
    dataset_tokenizer = datasets["train"]
    if args.model_type == "llama":
        # https://github.com/timinar/BabyLlama/blob/main/train.py
        tokenizer = ByteLevelBPETokenizer(add_prefix_space=True,trim_offsets=True)
        # tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        # tokenizer.decoder = decoders.ByteLevel()
        # tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        tokenizer.normalizer = NFKC()

        tokenizer.train_from_iterator(batch_iterator(), vocab_size=16000, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
        ])

        # Save files to disk
        tokenizer.save_model(model_path)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path, max_len=512)
        tokenizer.model_max_length = 128
        tokenizer.bos_token = "<s>"
        tokenizer.eos_token = "</s>"
        tokenizer.pad_token = "<pad>"      
    else:
        # https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling#train-tokenizer
        tokenizer = ByteLevelBPETokenizer()
        # Customized training
        tokenizer.train_from_iterator(batch_iterator(), vocab_size=52_000, min_frequency=2, special_tokens=[
            "<s>",
            "<pad>",
            "</s>",
            "<unk>",
            "<mask>",
        ])

        # Save files to disk
        tokenizer.save_model(model_path)
        tokenizer = RobertaTokenizerFast.from_pretrained(model_path, max_len=512)

# Setup custom data_collator:
#   we still use dynamic masking (mask differently at each epoch) as in the original RoBERTa paper, but do so deterministically (by setting the torch seed based on a hash of the document and epoch).
#       this is done to make the influence estimation more realistic
#   Aside: we do not use sentence packing as that would defeat the purpouse of applying an influence estimation method on a per-document basis
data_collator = None
if "roberta" in args.model_type:
    data_collator = util.DeterministicDataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )
    data_collator.set_epoch(0)
else:
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

#   we need to change some classes so that they pass down the current epoch to this datacollator, as well as to the dataloader:


class EpochVariableDataLoader(DataLoader):
    """A version of DataLoder that passes trough the epoch to a specified function when it's set_epoch is called.
       To enable deterministic dynamic masking, the current epoch must be passed down to the data_collator but the Trainer only calls DataLoader.set_epoch().
    """
    def __init__(self, train_dataset, passtrough_function, **dataloader_params):
        self.passtrough_function = passtrough_function
        super().__init__(train_dataset, **dataloader_params)
    def set_epoch(self, epoch):
        self.sampler.epoch = epoch    
        self.passtrough_function(epoch)    
 

class OrderedSampler(SequentialSampler):
    """Loads the curriculum from config["curriculum_path"]:
       This file is either a tensor of shape (num_epochs, n), where each row is treated as an epoch, or a list of tensors where each element is treated as an epoch. 
       The curriculum (and the dataset) may vary in lenght by epoch. 
       *The huggingface Trainer is oblivious of this so keep that in mind when looking at tqdm runtime estimates!*
    """
    def __init__(self, data_source, epoch):
        self.data_source = data_source
        self.epoch = epoch
        self.curriculum = util.get_curriculum(args.dataset, args.curriculum)
       
    def __iter__(self):
        return iter(self.curriculum[self.epoch].tolist())

# load and pre-tokenize the dataset (TODO unclear if that actually increases peformance with our custom dataloader and datacollator)

def tokenize(x):
    result = tokenizer(x["text"], return_special_tokens_mask=True, truncation=True, max_length=512)

    # hotfix: some curriculum creation strategies cause batches made exclusively from empty strings in rare edge cases
    # prevent error in hf implementation by just returning a pad token
    if (len(result["input_ids"]) == 0) and ("random" not in args.curriculum):
        return tokenizer("<pad>", return_special_tokens_mask=True, truncation=True, max_length=512)
    else:
        return result

dataset = datasets["train"]
dataset = dataset.map(tokenize)
dataset = dataset.remove_columns(["text"]) 
dataset.set_format("torch")

dataset_eval = datasets["validation"]
dataset_eval = dataset_eval.map(tokenize)
dataset_eval = dataset_eval.remove_columns(["text"]) 
dataset_eval.set_format("torch")



class CurriculumTrainer(Trainer):
    def get_train_dataloader(self) -> DataLoader:
        """
        Adapted to use EpochVariableDataLoader defined below
        Facilitates passing current epoch down to the data_collator (for deterministic dynamic masking) and dataloader (for loading the correct stage in the curriculum)
        Skips accelerator (as that just re-instantiates with the default DataLoader class)!
        """
        train_dataset = self.train_dataset
        data_collator = self.data_collator

        train_dataset = self._remove_unused_columns(train_dataset, description="training")
        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = OrderedSampler(self.train_dataset, self.state.epoch if self.state.epoch is not None else 0)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return EpochVariableDataLoader(train_dataset, data_collator.set_epoch if args.model_type == "roberta" else lambda _:None , **dataloader_params) # the Trainer class calls set_epoch on the dataloader, but we also need it in the data_collator


# set up eval 
from collections import defaultdict
batch_metrics = defaultdict(lambda:0) 
def compute_metrics(eval_pred, compute_result=True):
    """Computes accuracy as in https://github.com/huggingface/transformers/blob/main/examples/flax/language-modeling/run_mlm_flax.py 

    Args:
        eval_pred: Tuple of logits and labels
        compute_result: Trainer will set this to true once all batches are complete. Defaults to True.

    Returns:
        Metrics that are logged to W&B
    """
    global batch_metrics 
    logits, labels = eval_pred
    if not torch.is_tensor(logits):
        logits = torch.tensor(logits)
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
   
    predictions = torch.argmax(logits, axis=-1)
    label_mask = torch.where(labels > 0, 1.0, 0.0)

    batch_metrics["accuracy"] += ((torch.equal(predictions, labels))* label_mask).sum()
    
    # batch_metrics["mlm_loss"] += (torch.nn.functional.cross_entropy(logits, torch.nn.functional.one_hot((labels*label_mask).to(torch.int64), logits.shape[-1]).to(torch.float64))* label_mask).sum()
    batch_metrics["normalizer"] += label_mask.sum() # number of non-masked labels, divide this when compute_result to get mean 

    if compute_result:
        result = {
            "accuracy": batch_metrics["accuracy"] / batch_metrics["normalizer"],
            "normalizer" : batch_metrics["normalizer"]
            }
        batch_metrics = defaultdict(lambda:0) 
        return result
    else:
        return {}

# configs
roberta_config = RobertaConfig(
    vocab_size=tokenizer.vocab_size,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=12,
    type_vocab_size=1,
    layer_norm_eps=1e-05,
    attention_probs_dropout_prob = 0.1,
    hidden_act = "gelu",
    hidden_dropout_prob=0.1,
    hidden_size =768,
    initializer_range=0.02,
    intermediate_size=3072,
)

# https://github.com/timinar/BabyLlama/blob/main/config/llama-97M-strict.yaml
llama_config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=2*tokenizer.model_max_length,
        hidden_size=768,
        intermediate_size=2048,
        num_hidden_layers=12,
        num_attention_heads=12,
        tie_word_embeddings=True,
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    )

EPOCHS = len(util.get_curriculum(args.dataset, args.curriculum)) # note that an epoch is not necesarilly a pass over the entire dataset anymore
print("EPOCHS",EPOCHS)

training_args = TrainingArguments(
    seed=42,
    output_dir=model_path,
    save_strategy="epoch",
    overwrite_output_dir=True,

    num_train_epochs=EPOCHS, # do not change this manually: see the custom OrderedSampler  
    dataloader_num_workers=10,
    fp16=False, # TODO was True in https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/config/pretraining/base.yaml but loss extreme at start
    prediction_loss_only=False,
    remove_unused_columns=True,

    # https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md
    # for an effective batch size of  2048=16*64* 2 GPUS:
    #                                 2048=16*32* 4 GPUS
        per_device_train_batch_size=32,
        gradient_accumulation_steps=16,
        learning_rate=5e-4 if args.model_type == "roberta" else 7e-4, 

        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_epsilon=1e-06,
        weight_decay=0.01,
        lr_scheduler_type="polynomial" if args.model_type == "roberta" else "cosine",
        warmup_steps=10000 if args.model_type == "roberta" else 300, 
    # eval
        eval_strategy="epoch" if EPOCHS <=10 else "steps",
        eval_steps=None if EPOCHS <=10 else 0.1,
        label_names=["labels"], # of eval_dataset
        batch_eval_metrics=True,
        per_device_eval_batch_size=128,
        eval_on_start = False,

    # logging
        report_to="wandb", 
        logging_steps=50,   

    # debug
        use_cpu=False,
)
model = None
if "roberta" == args.model_type:
    model = RobertaForMaskedLM(config=roberta_config)
else:
    model = LlamaForCausalLM(llama_config)
trainer = CurriculumTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=dataset_eval,
    compute_metrics=compute_metrics,
    )
trainer.train()  
trainer.save_model(model_path)


########
model_name = os.path.basename(model_path)
model.push_to_hub(model_name, private=True)
tokenizer.push_to_hub(model_name, private=True)

from huggingface_hub import HfApi
from huggingface_hub import upload_folder
api = HfApi()
from util import get_epoch_checkpoints

for checkpoint_path in get_epoch_checkpoints(model_path):

    upload_folder(
        folder_path=checkpoint_path,
        path_in_repo="checkpoints/" + os.path.basename(checkpoint_path),
        repo_id=api.whoami()["name"] + "/" + model_name,
        repo_type="model",
        create_pr=False,
    )