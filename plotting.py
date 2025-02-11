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




def plot_per_document_in_order(df, curriculum):
    if isinstance(curriculum, list):
        curriculum = pad_sequence(curriculum,padding_value=-1).T
        
    dff = df[list(df.columns[0:-5])]


    out = np.empty((len(curriculum), len(curriculum[0])))
    for i in range(0, len(curriculum)):
        # print(i // len(dff.columns))
        out[i,:] = dff.iloc[curriculum[i].numpy(),i // len(dff.columns)]
        out[i,curriculum[i].numpy() == -1] = np.nan


    import matplotlib.pyplot as plt

    fig = plt.gcf()
    fig.set_size_inches(18.5, len(curriculum)*0.25)



    z = 3
    plt.imshow(out,aspect="auto",interpolation="none", cmap="PiYG",
            vmin= np.nanmean(out) - z*np.nanstd(out),
            vmax= np.nanmean(out) + z*np.nanstd(out)
            )


    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_scientific(False)

    ax.set_xticks([], [], minor=False)

    ax.set_yticks(np.arange(0.5, len(curriculum), 1), [], minor=False)
    ax.set_yticks(np.arange(0.0, len(curriculum), 1), ["At end of epoch {}".format(i) for i in np.arange(0, len(curriculum))], minor=True)
    ax.grid(color='w', linestyle='-', linewidth=1)
    plt.title("Per document in order")




def plot_per_document_per_stage(df):
    raise NotImplementedError # TODO update as above
    dff = df[list(df.columns[0:-5])+["stage"]].sort_values(by="stage")
    # dff[dff.columns[0:-1]] = dff[dff.columns[0:-1]].apply(replace_outliers_with_nan)

    vc = dff.groupby(by="stage")["stage"].count()
    counts = vc.to_numpy()
    sources = vc.index.tolist()
    ranges = [(start,stop) for start, stop in zip (np.cumsum(np.hstack((0,counts))),np.cumsum(counts))]
    x_ticks,x_ticks_end = zip(*ranges)
    out = np.empty((len(df.columns[0:-5]), len(df)))
    for i, (source, (start,stop)) in enumerate(zip(sources, ranges)):
        # print(source, start, stop)
        # display(df[df["source"] == source][df.columns[0:-4]].to_numpy().T.shape)
        out[:,start:stop] = dff[dff["stage"] == source][dff.columns[0:-1]].to_numpy().T#.reshape(1, len(df))
    import matplotlib.pyplot as plt

    fig = plt.gcf()
    fig.set_size_inches(18.5, 5)


    z = 3
    plt.imshow(out,aspect="auto",interpolation="none", cmap="PiYG",
            vmin= out.mean() - z*out.std(),
            vmax= out.mean() + z*out.std()
            )
    ax = plt.gca()
    ax.get_xaxis().get_major_formatter().set_scientific(False)


    ax.set_xticks(x_ticks, [], minor=False)
    ax.set_xticks(np.array(x_ticks_end) - ((np.array(x_ticks_end)-np.array(x_ticks))/2)  , sources, minor=True, rotation=45)

    ax.set_yticks(np.arange(0.5, len(dff.columns[0:-1]), 1), [], minor=False)
    ax.set_yticks(np.arange(0.0, len(dff.columns[0:-1]), 1), ["At end of epoch {}".format(i) for i in np.arange(0, len(dff.columns[0:-1]))], minor=True)
    ax.grid(color='w', linestyle='-', linewidth=1)

    # ax.get_yaxis().get_major_formatter().set_scientific(False)

    plt.title("Per document, per stage/domain")

def plot_per_token_per_stage(df):
    dff = df[list(df.columns[0:-5])+["stage", "document_lenght"]].sort_values(by="stage")
    vc = dff.groupby(by="stage")["document_lenght"].sum()
    counts = vc.to_numpy()
    sources = vc.index.tolist()
    ranges = [(start,stop) for start, stop in zip (np.cumsum(np.hstack((0,counts))),np.cumsum(counts))]




    out = np.empty((len(df.columns[0:-5]), df["document_lenght"].sum()))
    for i, (source, (start,stop)) in enumerate(zip(sources, ranges)):
        # print(source, start, stop)
        # display(df[df["source"] == source][df.columns[0:-4]].to_numpy().T.shape)
        a = df[df["stage"] == source][df.columns[0:-5]].to_numpy().T
        dl = df[df["stage"] == source]["document_lenght"].to_numpy().T
        
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

    ax.set_yticks(np.arange(0.5, len(df.columns[0:-5]), 1), [], minor=False)
    ax.set_yticks(np.arange(0.0, len(df.columns[0:-5]), 1), ["At end of epoch {}".format(i) for i in np.arange(0, len(df.columns[0:-5]))], minor=True)
    ax.grid(color='w', linestyle='-', linewidth=1)

    # ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.title("Per token ([doc]* tokes_doc)")
def plot_per_token_in_order(df, curriculum):
    dff = df[list(df.columns[0:-5])+[ "document_lenght"]]
    if isinstance(curriculum, list):
        curriculum = pad_sequence(curriculum).T


    out = np.empty((len(df.columns[0:-5]), df["document_lenght"].sum()))
    for i in range(0, len(df.columns[0:-5])):
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
    ax.get_xaxis().get_major_formatter().set_scientific(False)



    ax.set_xticks([], [], minor=False)


    ax.set_yticks(np.arange(0.5, len(df.columns[0:-5]), 1), [], minor=False)
    ax.set_yticks(np.arange(0.0, len(df.columns[0:-5]), 1), ["At end of epoch {}".format(i) for i in np.arange(0, len(df.columns[0:-5]))], minor=True)
    ax.grid(color='w', linestyle='-', linewidth=1)

    # ax.get_yaxis().get_major_formatter().set_scientific(False)
    plt.title("In order, per token ([doc]* tokes_doc)")
