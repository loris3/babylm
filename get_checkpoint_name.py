from dotenv import load_dotenv
load_dotenv()


import argparse


from util import get_checkpoints_hub


import os


parser = argparse.ArgumentParser("")
parser.add_argument("model", help="A model on the hf hub. Format: username/name_curriculum")
parser.add_argument("checkpoint_nr", help="Id of the checkpoint to extract gradients for (starting at 0)",type=int)

args = parser.parse_args()


if __name__ == "__main__":
    print(os.path.basename(get_checkpoints_hub(args.model)[args.checkpoint_nr]))
