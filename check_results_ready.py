import argparse
import os

from dotenv import load_dotenv
load_dotenv()



parser = argparse.ArgumentParser("check_results_ready")

parser.add_argument("model", help="A model on the hf hub. Format: username/name_curriculum")
parser.add_argument("dataset_train", help="A dataset on the hf hub. Format: username/name")
parser.add_argument("--dataset_train_split", help="The split to access", default="train")
parser.add_argument("checkpoint_nr", help="Id of the checkpoint to extract gradients for (starting at 0)",type=int)
parser.add_argument("--num_processes", help="Number of processes to use when doing dot product (runs on cpu)", type=int, nargs="?", const=1, default=1)
parser.add_argument("--gradients_per_file", help="Number of gradients per output file", type=int, nargs="?", const=1, default=1000)
parser.add_argument("--batch_size", help="How many chunks each subprocess will keep in memory", type=int, nargs="?", const=1, default=200)

parser.add_argument("--mode", help="If 'mean', mean influence of individual train on all examples in test; if 'single' 1 train -> 1 test", default="single")
parser.add_argument("--dataset_test", help="A dataset on the hf hub. If supplied, returns one score per test instance. Format: username/name", default=None)
parser.add_argument("--dataset_test_split", help="The split to access", default="test")

parser.add_argument("--delete_test_on_success", default=False, action='store_true')
parser.add_argument("--delete_train_on_success", default=False, action='store_true')

parser.add_argument("--test", help="The split to access", default=False)
parser.add_argument("--test_dataset_size", help="The split to access", default=0, type=int)
args = parser.parse_args()


from datasets import load_dataset

model_name = args.model.split("/")[-1]
dataset_train_name = args.dataset_train.split("/")[-1]
dataset_train_split_name = args.dataset_train_split
dataset_test_name = args.dataset_test.split("/")[-1]
dataset_test_split_name = args.dataset_test_split


influence_dir = "./influence" if args.mode == "single" else "./mean_influence"

if not os.path.exists(influence_dir):
    os.makedirs(influence_dir)
    
influence_output_dir = os.path.join(influence_dir, model_name, "_".join([dataset_train_name, dataset_train_split_name, dataset_test_name, dataset_test_split_name]))
if not os.path.exists(influence_output_dir):
    os.makedirs(influence_output_dir)



import os





from util import get_checkpoints_hub
checkpoints =  get_checkpoints_hub(args.model) if not args.test else ["test/test/test/test"]






if __name__ == '__main__':   
    checkpoint = checkpoints[args.checkpoint_nr]
    out_path = os.path.join(influence_output_dir, os.path.basename(checkpoint))
    print(os.path.isfile(out_path))

