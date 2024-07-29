from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
import math
import os
from vllm import LLM
import random
from transformers import set_seed, AutoTokenizer
import os
import pandas as pd
import time
import numpy as np
import sys
import torch
from datasets import load_dataset, Dataset
from sentence_transformers import SentenceTransformer
import mteb
from mteb.models.text_formatting_utils import corpus_to_texts

from run_mteb_on_datasets import load_model


set_seed(123456)


def shard_dict_corpus(corpus: list[dict[str, str]] | dict[str, list[str]], shard_total: int, shard_index: int):
    # Sort the keys to ensure consistent sharding
    sorted_keys = sorted(corpus.keys())
    shard_size = math.ceil(len(sorted_keys) / shard_total)
    start_index = shard_index * shard_size
    end_index = min((shard_index + 1) * shard_size, len(sorted_keys))
    
    shard_keys = sorted_keys[start_index:end_index]
    shard_corpus = {key: corpus[key] for key in shard_keys}
    print(f"For shard {shard_index}, using start index {start_index} and end index {end_index}")
    
    return shard_corpus


def embed_dataset(args, corpus: list[dict[str, str]] | dict[str, list[str]], model):
    ids = []
    docs = []
    for k, v in corpus.items():
        ids.append(k)
        docs.append(v)
    # Start the multi-process pool on all available CUDA devices

    # save embeddings to file, at nfs/{dataset_name}/{model_name}.npz
    # remove the file if it existed before      
    file_safe_model = args.model.replace("/", "_")
    file_safe_dataset = args.dataset.replace("/", "--")

    file_name = f'indexes/{file_safe_dataset}-{args.split}/{file_safe_model}/embedding_{args.shard}--{args.shard_total}.jsonl'
    if not os.path.isdir(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    if os.path.isfile(file_name):
        os.remove(file_name)
    
    outputs = model.encode_corpus(docs)

    sentences = corpus_to_texts(docs)

    file_to_write = open(file_name, "w")
    total = 0
    for idx, item in enumerate(outputs):
        # save `id` and `embeddings` for the index and `text` if we want to use it for queries
        file_to_write.write(json.dumps({"id": ids[idx], "vector": item.tolist(), "text": sentences[idx]}) + "\n")
        total += 1

    file_to_write.close()
    dim_size = outputs.shape[1]
    print(f"Embedding dimension: {dim_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed a dataset")
    parser.add_argument("--model", type=str, help="The model name", default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument("--dataset", type=str, help="The dataset name", default="NQ")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size")
    parser.add_argument("--split", type=str, default="test", help="The split to embed")
    parser.add_argument("--subsplit", type=str, default=None, help="The subsplit to embed")
    parser.add_argument("--shard", type=int, default=0, help="The shard to embed")
    parser.add_argument("--shard_total", type=int, default=1, help="The total number of shards")
    args = parser.parse_args()

    # Load the dataset
    tasks = mteb.get_tasks(tasks=[args.dataset])
    assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"
    tasks[0].load_data()
    corpus = tasks[0].corpus[args.split]
    if args.subsplit:
        corpus = corpus[args.subsplit]

    ## keep only the args.shard part of args.shard_total, using np to split into shard_total parts
    corpus = shard_dict_corpus(corpus, args.shard_total, args.shard)
    print(f"Shard {args.shard} has {len(corpus)} documents")

    model = mteb.get_model(args.model)

    # Embed the dataset
    embed_dataset(args, corpus, model)