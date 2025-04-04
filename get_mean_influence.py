import torch
import os
import argparse
from datasets import load_dataset

parser = argparse.ArgumentParser("mean_normalized_gradient_per_category")

parser.add_argument("model", help="A model on the hf hub. Format: username/name_curriculum")
parser.add_argument("dataset_train", help="A dataset on the hf hub. Format: username/name")
parser.add_argument("--dataset_train_split", help="The split to access", default="train")
parser.add_argument("--category_key", default="noise_type")
parser.add_argument("--checkpoint_name", default="main")
parser.add_argument("--gradients_per_file", help="Number of gradients per output file", type=int, nargs="?", const=1, default=1000)

parser.add_argument("--proj_dim", type=int, nargs="?", const=1, default=2**14)


args = parser.parse_args()#args=["allenai/OLMo-2-1124-7B-SFT", "allenai/tulu-3-sft-mixture","--category_key=source"])

gradient_dir = os.path.join("./gradients", str(args.proj_dim), os.path.basename(args.model), os.path.basename(args.dataset_train))

dataset = load_dataset(args.dataset_train, split=args.dataset_train_split)
len_dataset = len(dataset)


chunks_train = [[ (i,min(i+args.gradients_per_file, len_dataset), os.path.join(gradient_dir, chunk, os.path.basename(args.checkpoint_name), str(i) + "_" + str(i + args.gradients_per_file))) for i in range(0, len_dataset, args.gradients_per_file)] for chunk in [args.dataset_train_split + f"[{i}%:{i + 100 // 100}%]" for i in range(0, 100, 100 // 100)]]
chunks_train = [item for sublist in chunks_train for item in sublist]
# chunks_train = [(start,stop,chunk) for start,stop,chunk in chunks_train if os.path.exists(chunk)]

shape = torch.load(chunks_train[0][-1]).shape


sums = {k: torch.zeros((1,1,shape[-1]),dtype=torch.float64) for k in dataset.unique(args.category_key)}
for start, stop, path in chunks_train:
    try:
        gradients = torch.nn.functional.normalize(torch.load(path), dim=1)
        for source, gradient in zip(dataset[args.category_key][start:stop],gradients):
            sums[source] = sums[source] + gradient
    except:
        print(path, "missing!")
        raise FileNotFoundError


for category, summed_gradient in sums.items():
    mean_normalized_gradient = torch.nn.functional.normalize(summed_gradient / len_dataset) # normalize mean of normalized gradients to get (newly) normalized gradient
    out_path = os.path.join("./influence_mean_normalized", os.path.basename(args.model), os.path.basename(args.dataset_train), category, args.checkpoint_name)
    torch.save(mean_normalized_gradient, out_path)
