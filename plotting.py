import os

from dotenv import load_dotenv
load_dotenv()
import datasets
from datasets import DatasetDict
from datasets import load_dataset
import datasets
import torch
import json
from huggingface_hub import HfApi
import numpy as np


from scipy.stats import norm
from scipy.stats import f
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import torch
import datasets
from datasets import load_dataset
import util
from scipy.stats import zscore
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
def load_data_for_plotting(dataset_name, dataset_split, model_name, curriculum_name="", influence_output_dir="./mean_influence"):
    # model_name = os.path.join(os.path.basename(dataset_name)+"_"+("" if model_type == "" else model_type +"_")+curriculum_name.split(".")[0])
    influence_output_dir = os.path.join(influence_output_dir, os.path.basename(model_name), os.path.basename(dataset_name) + "_" +dataset_split + "_" + os.path.basename(dataset_name) + "_" +dataset_split)
    print("influence_output_dir",influence_output_dir)
    dataset = load_dataset(dataset_name)["train"]

  

    df = pd.DataFrame({result_checkpoint: torch.load(os.path.join(influence_output_dir,result_checkpoint),weights_only=True,map_location="cpu").numpy().flatten() for result_checkpoint in os.listdir(influence_output_dir)})
    df.sort_index(axis=1)

    df = df.reindex(sorted(df.columns, reverse=False), axis=1)
    influence_cols = df.columns
    df["total"] = df.sum(axis=1)
    if "text" in dataset.features:
        df[["text", "source","stage"]] = dataset.to_pandas()
        df["document_lenght"] = df["text"].str.split().str.len()
    else:
        df[["output", "input","instruction"]] = dataset.to_pandas() # TODO hotfix
    return df, util.get_curriculum(dataset_name, curriculum_name) if curriculum_name != "" else None




def plot_per_document_in_order(dff, curriculum):
    if isinstance(curriculum, list):
        curriculum = pad_sequence(curriculum,padding_value=-1).T
        


    out = np.empty((len(dff.columns), len(curriculum[0])))
    for i in range(0, len(dff.columns)):
   
        out[i,:] = dff.iloc[curriculum[i].numpy(),i]
        out[i,curriculum[i].numpy() == -1] = np.nan



    fig = plt.gcf()
    fig.set_size_inches(18.5, len(curriculum)*0.25)



    z = 3
    plt.imshow(out,aspect="auto",interpolation="none", cmap="PiYG",
            vmin= np.nanmean(out) - z*np.nanstd(out),
            vmax= np.nanmean(out) + z*np.nanstd(out)
            )


    ax = plt.gca()
   # ax.get_xaxis().get_major_formatter().set_scientific(False)

    ax.set_xticks([], [], minor=False)

    ax.set_yticks(np.arange(0.5, len(curriculum), 1), [], minor=False)
    ax.set_yticks(np.arange(0.0, len(curriculum), 1), ["At end of epoch {}".format(i) for i in np.arange(0, len(curriculum))], minor=True)
    ax.grid(color='w', linestyle='-', linewidth=1)
    plt.title("Per document in order")




import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

def plot_per_document_per_stage(dff, curriculum):
    if isinstance(curriculum, list):
        curriculum = pad_sequence(curriculum, padding_value=-1).T
        
    dff = dff.sort_values(by="stage")
    
    vc = dff.groupby("stage")["stage"].count()
    counts = vc.to_numpy()
    sources = vc.index.tolist()
    ranges = [(start, stop) for start, stop in zip(np.cumsum(np.hstack((0, counts))), np.cumsum(counts))]
    x_ticks, x_ticks_end = zip(*ranges)
    
    out = np.empty((len(dff.columns) - 1, len(dff)))
    for i, (source, (start, stop)) in enumerate(zip(sources, ranges)):
        out[:, start:stop] = dff[dff["stage"] == source][dff.columns[:-1]].to_numpy().T
    
    fig, ax = plt.subplots(figsize=(18.5, 5))
    
    z = 3
    im = ax.imshow(out, aspect="auto", interpolation="none", cmap="PiYG",
                   vmin=out.mean() - z * out.std(),
                   vmax=out.mean() + z * out.std())
    
    for pos in x_ticks_end[:-1]:
        ax.axvline(x=pos - 0.5, color='black', linestyle='--', linewidth=1)
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([])
    ax.set_xticks(np.array(x_ticks_end) - ((np.array(x_ticks_end) - np.array(x_ticks)) / 2), minor=True)
    ax.set_xticklabels(sources, minor=True, rotation=45)
    
    ax.set_yticks(np.arange(0.5, len(dff.columns) - 1, 1))
    ax.set_yticks(np.arange(0.0, len(dff.columns) - 1, 1), minor=True)
    ax.set_yticklabels(["At end of epoch {}".format(i) for i in np.arange(0, len(dff.columns) - 1)], minor=True)
    ax.grid(color='w', linestyle='-', linewidth=1)
    
    plt.colorbar(im, ax=ax, label="Value")
    plt.title("Per document, per stage/domain")
    plt.show()



def plot_per_token_per_stage(dff,curriculum):
    dff = dff.sort_values(by="stage")
    vc = dff.groupby(by="stage")["document_lenght"].sum()
    counts = vc.to_numpy()
    sources = vc.index.tolist()
    ranges = [(start,stop) for start, stop in zip (np.cumsum(np.hstack((0,counts))),np.cumsum(counts))]
   # print(ranges,ranges[-1][-1])



    out = np.empty((len(dff.columns)-2, ranges[-1][-1]))
    for i, (source, (start,stop)) in enumerate(zip(sources, ranges)):
        # print(source, start, stop)
        # display(df[df["source"] == source][dff.columns[0:-4]].to_numpy().T.shape)
        a = dff[dff["stage"] == source][dff.columns[0:-2]].to_numpy().T
       # print(a.shape,start,stop)
        dl = dff[dff["stage"] == source]["document_lenght"].to_numpy().T
        
        for j in range(0, a.shape[0]):
            out[j,start:stop] = np.repeat(a[j],dl, axis=0)



    import matplotlib.pyplot as plt

    chunk_size = 2  # Size of chunks
    new_length = out.shape[1] // chunk_size
    downsampled = out[:, :new_length * chunk_size].reshape(out.shape[0], new_length, chunk_size).mean(axis=2)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 5)


    out = downsampled
    z = 3
    plt.imshow(out,aspect="auto",interpolation="none", cmap="PiYG",
            vmin= out.mean() - z*out.std(),
            vmax= out.mean() + z*out.std()
            )



    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_scientific(False)

    x_ticks,x_ticks_end = zip(*[(start//chunk_size,stop//chunk_size) for start, stop in ranges])


    ax.set_xticks(x_ticks, [], minor=False)
    ax.set_xticks(np.array(x_ticks_end) - ((np.array(x_ticks_end)-np.array(x_ticks))/2)  , sources, minor=True, rotation=45)

    ax.set_yticks(np.arange(0.5, len(dff.columns[0:-2]), 1), [], minor=False)
    ax.set_yticks(np.arange(0.0, len(dff.columns[0:-2]), 1), ["At end of epoch {}".format(i) for i in np.arange(0, len(dff.columns[0:-2]))], minor=True)
    ax.grid(color='w', linestyle='-', linewidth=1)

    
    plt.title("Per token ([doc]* tokes_doc)")
    plt.show()
def plot_per_token_in_order(dff, curriculum):
    if isinstance(curriculum, list):
        curriculum = pad_sequence(curriculum).T


    out = np.empty((len(dff.columns[0:-2]), dff["document_lenght"].sum()))

    for i in range(0, len(dff.columns[0:-2])):
        # print(source, start, stop)
        # display(df[df["source"] == source][df.columns[0:-4]].to_numpy().T.shape)
        a = dff.iloc[curriculum[i,:].numpy(),i].to_numpy().T
        dl = dff["document_lenght"].iloc[curriculum[i,:].numpy()].to_numpy().T
        out[i,:] = np.repeat(a,dl, axis=0)



    import matplotlib.pyplot as plt

    chunk_size = 2  # Size of chunks
    new_length = out.shape[1] // chunk_size
    downsampled = out[:, :new_length * chunk_size].reshape(out.shape[0], new_length, chunk_size).mean(axis=2)

    fig = plt.gcf()
    fig.set_size_inches(18.5, 5)


    out = downsampled
    z = 3
    plt.imshow(out,aspect="auto",interpolation="none", cmap="PiYG",
            vmin= out.mean() - z*out.std(),
            vmax= out.mean() + z*out.std()
            )



    ax = plt.gca()
    # ax.get_xaxis().get_major_formatter().set_scientific(False)



    ax.set_xticks([], [], minor=False)


    ax.set_yticks(np.arange(0.5, len(dff.columns[0:-2]), 1), [], minor=False)
    ax.set_yticks(np.arange(0.0, len(dff.columns[0:-2]), 1), ["At end of epoch {}".format(i) for i in np.arange(0, len(dff.columns[0:-2]))], minor=True)
    ax.grid(color='w', linestyle='-', linewidth=1)

    # ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.title("In order, per token ([doc]* tokes_doc)")
    plt.show()