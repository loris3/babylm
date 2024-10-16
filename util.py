from pathlib import Path
import os
def get_epoch_checkpoints(model_dir):
    c = sorted([int(str(x).split("-")[-1]) for x in Path(model_dir).glob("checkpoint-*")])
    epoch_checkpoints = c[5::6]
    if c[-1] not in epoch_checkpoints:
        epoch_checkpoints.pop()
        epoch_checkpoints.append(c[-1])
    return epoch_checkpoints

def get_all_chunks(checkpoint_path,gradient_input_dir, gradients_per_file):
    return [ os.path.join(gradient_input_dir, checkpoint_path.split("-")[-1] + "_" + str(i) + "_" + str(i + gradients_per_file)) for i in range(0, len(dataset["train"]), args.gradients_per_file)]

