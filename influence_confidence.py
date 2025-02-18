
import argparse
import os


from dotenv import load_dotenv
load_dotenv()

import wandb

import torch
import logging

from datasets import load_dataset
import time
import torch 

from util import get_checkpoints_hub

from transformers import  DataCollatorForLanguageModeling
from transformers import AutoModelForCausalLM, AutoTokenizer
 

os.environ["TOKENIZERS_PARALLELISM"] = "True"

parser = argparse.ArgumentParser("gradient_extraction")
parser.add_argument("--model", help="A model on the hf hub. Format: username/name", default="allenai/OLMo-2-1124-7B-SFT")
parser.add_argument("--dataset", help="A dataset on the hf hub. Format: username/name", default="yuxixia/triviaqa-test-tulu3-query")
parser.add_argument("--dataset_split", help="The split to access", default="train[0%:100%]")
parser.add_argument("--checkpoint_nr", help="Id of the checkpoint to extract gradients for (inferred from the repo, starting at 0)",type=int, default=0)
parser.add_argument("--output_path", help="The path where to store the result dataset at", default="./results_confidence")

parser.add_argument("--batch_size", help="How many examples are assigned to each process (lifetime. no re-use, mainly affects ram requirements)", type=int, default=10000)
parser.add_argument("--num_proc", help="How many processes to use (one model loaded per process, increase until OOM)", type=int, default=4)
args = parser.parse_args()


model_name = args.model.split("/")[-1]
dataset_name = args.dataset.split("/")[-1]
dataset_split_name = args.dataset_split


logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

tokenizer = AutoTokenizer.from_pretrained(args.model)
dataset = load_dataset(args.dataset, split=args.dataset_split) 
checkpoints =  get_checkpoints_hub(args.model)
data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )










if __name__ == '__main__':
    from multiprocess import set_start_method
    set_start_method("spawn")
    device = "cuda"

    
    
    run = wandb.init(project="confidence_gradient_extraction")
    run.name = os.path.join(args.model, os.getenv("SLURM_JOB_NAME", "?"))
    #logging.info("Getting gradients for checkpoint-{}".format(checkpoint))

  
    

    start_time = time.time()
   
    



    

    def get_influence_batch(ds, rank):
        from dotenv import load_dotenv
        load_dotenv()
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % torch.cuda.device_count())

        print(os.environ["CUDA_VISIBLE_DEVICES"])


        model = AutoModelForCausalLM.from_pretrained(args.model, revision="main", torch_dtype=torch.float16, device_map="cuda")
        model.train()

        device = "cuda"


        def preprocess_tulu_confidence_batched(data_samples):
            questions = data_samples["question"]
            responses = data_samples["response"]
            batched_messages = [
                [
                    {"role": "user", "content": f"Answer the question, give ONLY the answer, no other words or explanation:\n\n" + question},
                        {"role": "assistant", "content":response},
                        {"role": "user", "content": f"Provide the probability that your answer is correct. Give ONLY the probability between 0.0 and 1.0, no other words or explanation."}
                ]
                for question, response in zip(questions, responses) 
            ]
            
            return [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in batched_messages]


        def get_loss_gradient(model, example,device):
            """Computes gradient of the loss function irt to the input embeddings for MLM.

            Args:
                model: A model with a `forward` method wich returns `loss`
                example: An instance from the training data
                device: What GPU to use (e.g., cuda:0)

            Returns:
                A 1D tensor: gradient of the loss function irt to the input embeddings
            """
            
            model.zero_grad()
            

            input_ids, labels = data_collator(example).values() # the data_collator used here applies the exact same mask used in the respective epoch
            inputs_embeds=model.get_input_embeddings().weight[input_ids].to(device)
            inputs_embeds.retain_grad()

            outputs = model.forward(
                    inputs_embeds=inputs_embeds,
                    labels=labels.to(device)
                )
            loss = outputs.loss
            loss.retain_grad()
            return torch.autograd.grad(loss, inputs_embeds, retain_graph=False)[0].squeeze()
            
        
        def process_row(row):
            z_train = tokenizer(preprocess_tulu_confidence_batched(row)[0],return_special_tokens_mask=False, truncation=True, padding="max_length", max_length=4096,return_tensors="pt",padding_side='left')["input_ids"]
            grad_z_train = get_loss_gradient(model, z_train,device).detach().flatten().to(torch.float32)#.cpu()
            test_cols = ["confidence_query", "response_query"]
            for test_col in test_cols:
                row[test_col+"_influence"] = torch.empty(len(row[test_col]))

                examples = [list(row.values())[-1]["text"] for row in eval(row[test_col])] # TODO hotfix
                for i, example in enumerate(examples):
                    z_test = tokenizer(example,return_special_tokens_mask=False, truncation=True, padding="max_length", max_length=4096,return_tensors="pt",padding_side='left')["input_ids"]
                    grad_z_test = get_loss_gradient(model, z_test,device).detach().flatten().to(torch.float32)#.cpu()
                    row[test_col+"_influence"][i] = torch.dot(grad_z_train, grad_z_test).cpu()
                row[test_col+"_influence"] = row[test_col+"_influence"].numpy().tolist()
                
            return row
        
        return ds.apply(process_row, axis=1)
    

    
    dataset = dataset.with_format("pandas")
  
    result = dataset.map(get_influence_batch, num_proc=args.num_proc, with_rank=True,batched=True, batch_size=args.batch_size)
    result.save_to_disk(os.path.join("./results_confidence", dataset.info.dataset_name, "main"))
