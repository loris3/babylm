from pathlib import Path
import os
from huggingface_hub import hf_hub_download
def get_curriculum(repo_id , filename):
    return torch.load(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"),weights_only=True)



def get_epoch_checkpoints(model_dir):
    checkpoint_ids = sorted([int(str(x).split("-")[-1]) for x in Path(model_dir).glob("checkpoint-*")])

    return [os.path.join(model_dir,"checkpoint-{}".format(c)) for c in checkpoint_ids]
    # epoch_checkpoints = c[5::6]
    # if c[-1] not in epoch_checkpoints:
    #     epoch_checkpoints.pop()
    #     epoch_checkpoints.append(c[-1])
    return epoch_checkpoints
from huggingface_hub import snapshot_download

def get_checkpoints_hub(model):
    local_dir = snapshot_download(repo_id=model, allow_patterns=["checkpoints/*"])
    return [os.path.join(local_dir,"checkpoints", f) for f in os.listdir(os.path.join(local_dir, "checkpoints"))]


# def get_all_chunks(checkpoint_path,gradient_input_dir, gradients_per_file):
#     return [ os.path.join(gradient_input_dir, checkpoint_path.split("-")[-1] + "_" + str(i) + "_" + str(i + gradients_per_file)) for i in range(0, len(dataset["train"]), args.gradients_per_file)]
def get_epoch(checkpoint_path):
    checkpoint_ids = sorted([int(str(x).split("-")[-1]) for x in Path(os.path.dirname(checkpoint_path)).glob("checkpoint-*")])
    return checkpoint_ids.index(int(str(checkpoint_path).split("-")[-1]))


import xxhash

h = xxhash.xxh64()
def get_seed_for_document(document, epoch):
    """Returns the seed to be used to set torch.manual_seed when doing dynamic masking

    Args:
        document: A string to get the seed for
        epoch: The epoch to get the seed for

    Returns:
        An integer
    """
    h.update(document.cpu().numpy())
    h.update(bytes(epoch))
    seed = h.intdigest()
    h.reset()
    return seed

from transformers import DataCollatorForLanguageModeling
import torch
class DeterministicDataCollatorForLanguageModeling (DataCollatorForLanguageModeling): 
    def torch_mask_tokens(self, inputs, special_tokens_mask = None):
        """
        Adapted to make dynamic masking determinsitic based on (text, epoch). 
        Just wrapped the original implementation in a for loop where a seed based on (labels, epoch) is set for each individual example before masking.
        """
        labels = inputs.clone()

        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        for i in range(0, labels.shape[0]):
            torch.manual_seed(get_seed_for_document(labels[i], self.epoch))

            # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)

            probability_matrix = torch.full(labels[i:i+1].shape, self.mlm_probability)


           
            probability_matrix.masked_fill_(special_tokens_mask[i:i+1], value=0.0)
            masked_indices = torch.bernoulli(probability_matrix).bool()
            labels[i:i+1][~masked_indices] = -100  # We only compute loss on masked tokens

            # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
            indices_replaced = torch.bernoulli(torch.full(labels[i:i+1].shape, 0.8)).bool() & masked_indices
            inputs[i:i+1][indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            # 10% of the time, we replace masked input tokens with random word
            indices_random = torch.bernoulli(torch.full(labels[i:i+1].shape, 0.5)).bool() & masked_indices & ~indices_replaced
            random_words = torch.randint(len(self.tokenizer), labels[i:i+1].shape, dtype=torch.long)
            inputs[i:i+1][indices_random] = random_words[indices_random]

        ######################
        
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
    def set_epoch(self, epoch):
        self.epoch = epoch
