from transformers import RobertaTokenizerFast
import os

from nltk.lm import MLE
from datasets import load_dataset
import pandas as pd

from nltk.util import everygrams


def get_perplexity_ngram(dataset_name="loris3/stratified_10m_curriculum", n_gram=1, train_subsample_factor = 1):
    """
    Returns a training data order by increasing difficulty estimated with an ngram MLE model
    code from https://github.com/codebyzeb/CLIMB/blob/dc4f13c1bdf94938d468f0926368dee048485cba/src/data_curriculum/difficulty_scorer/perplexity.py#L77
    adapted to accept hf datasets
    """
    out_path = os.path.join("./difficulty_ngram",str(n_gram), os.path.basename(dataset_name))
    print(out_path)
    if os.path.isdir(out_path):
        print("Skipping {}, already calculated".format(out_path) )
    else:
        tokenizer = RobertaTokenizerFast.from_pretrained(dataset_name+"_random", max_len=512)

        dataset = load_dataset(dataset_name)["train"]
        dataset.set_transform(lambda x : tokenizer(x["text"], return_special_tokens_mask=True, truncation=True, max_length=512))
        train_data_n_grams = [
            list(everygrams(sent, max_len=n_gram))
            for sent in  
            (
                [str(id) for id in input_ids if id != tokenizer.pad_token_id]  # type: ignore
                    for input_ids in dataset[0 : dataset.num_rows : train_subsample_factor]["input_ids"]
            )
        ]
        train_vocab = [str(val) for val in tokenizer.vocab.values()]

        print("Fitting {}-gram model".format(n_gram))
        lm = MLE(n_gram)
        lm.fit(train_data_n_grams, train_vocab)
        print("Getting perplexity with {}-gram model".format(n_gram))
        df = pd.DataFrame([lm.perplexity(example) for example in train_data_n_grams])
        df.columns=["perplexity"]
        df.index.name = "document_id"
        os.makedirs(out_path)
        df.to_parquet(os.path.join(out_path,"perplexity"))
    return pd.read_parquet(os.path.join(out_path,"perplexity"))


import os
import en_core_web_sm
from lexical_diversity import lex_div as ld
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
from datasets import load_dataset  # Assuming you're using the HuggingFace datasets library



def mattr(docs):
    results = []

    nlp = en_core_web_sm.load()
    for doc in docs:
        l = [token.text.lower() for token in nlp(doc) if token.is_alpha]
        if len(l) > 0:
            results.append(ld.mattr(l, window_length=5))
        else:
            results.append(None)
    return results

def get_mattr(dataset_name="loris3/stratified_10m_curriculum", n_proc=24):
    out_path = os.path.join("./difficulty_mattr", os.path.basename(dataset_name))
    print(out_path)

    if os.path.isdir(out_path):
        print("Skipping {}, already calculated".format(out_path))
    else:

        data = load_dataset(dataset_name)["train"].to_pandas()["text"]
        results = []
        print("Getting mattr")

        chunk_size = len(data) // n_proc  
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

        with ProcessPoolExecutor(max_workers=n_proc) as executor:
            futures = [executor.submit(mattr, chunk) for chunk in chunks]
            for future in tqdm(as_completed(futures), total=len(futures)):
                results.extend(future.result()) 

        print("Saving mattr")
        df = pd.DataFrame(results, columns=["mattr"])
        df.index.name = "document_id"
        os.makedirs(out_path, exist_ok=True)
        df.to_parquet(os.path.join(out_path, "mattr"))

    return pd.read_parquet(os.path.join(out_path, "mattr"))
