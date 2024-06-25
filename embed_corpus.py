from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
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

from run_mteb_on_datasets import load_model


set_seed(123456)


def embed_dataset(args, dataloader: list, model, use_vllm: bool = False):
    # Start the multi-process pool on all available CUDA devices

    # save embeddings to file, at nfs/{dataset_name}/{model_name}.npz
    # remove the file if it existed before      
    file_safe_model = args.model.replace("/", "_")
    file_safe_dataset = args.dataset.replace("/", "--")

    file_name = f'indexes/{file_safe_dataset}/{file_safe_model}/embedding.jsonl'
    if not os.path.isdir(os.path.dirname(file_name)):
        os.makedirs(os.path.dirname(file_name))

    if os.path.isfile(file_name):
        os.remove(file_name)

    file_to_write = open(file_name, "w")
    total = 0
    # NOTE: I batch this to avoid having to hold all in memory, although it's not as efficient
    # because of this the models base batch size is high and this controls it
    for i, batch in enumerate(tqdm(dataloader)):
        # Compute the embeddings using the multi-process pool
        sentences = batch["text"]
        ids = batch["id"] if "id" in batch else batch["doc_id"]
        if use_vllm:
            outputs = model.encode(sentences)
            for idx, item in enumerate(outputs):
                # save `id` and `embeddings` for the index and `text` if we want to use it for queries
                file_to_write.write(json.dumps({"id": ids[idx], "vector": item.outputs.embedding, "text": sentences[idx]}) + "\n")
                total += 1
        else:
            batch_emb = model.encode(sentences, batch_size=args.batch_size, prompt_name=None)
            for idx, item in enumerate(batch_emb):
                # save `id` and `embeddings` for the index and `text` if we want to use it for queries
                file_to_write.write(json.dumps({"id": ids[idx], "vector": item.tolist(), "text": sentences[idx]}) + "\n")
                total += 1

    file_to_write.close()
    if use_vllm:
        dim_size = outputs.outputs.embedding.size(0)
    else:
        dim_size = batch_emb[0].shape[0]
    print(f"Embedding dimension: {dim_size}")



def combine_doc(doc: dict):
    if "title" in doc:
        return doc["title"] + " " + doc["text"]
    else:
        return doc["text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed a dataset")
    parser.add_argument("--model", type=str, help="The model name", default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument("--dataset", type=str, help="The dataset name", default="NQ")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size")
    parser.add_argument("--split", type=str, default="test", help="The split to embed")
    args = parser.parse_args()

    # Load the dataset
    tasks = mteb.get_tasks(tasks=[args.dataset])
    assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"
    tasks[0].load_data()
    corpus = tasks[0].corpus[args.split]

    input = [combine_doc(item) for item in corpus.values()]
    ids = list(corpus.keys())
    print(f"Dataset {args.dataset} has {len(input)} documents")

    # Load the model
    # if "mistral" in args.model:
    #     use_vllm = True
    #     model = LLM(model=args.model, enforce_eager=True, dtype="auto", disable_sliding_window=True)
    # else:
    use_vllm = False
    model = load_model(args.model, args.dataset)

    dataloader = DataLoader(Dataset.from_dict({"text": input, "id": ids}), batch_size=args.batch_size, num_workers=0)

    # Embed the dataset
    embed_dataset(args, dataloader, model, use_vllm=use_vllm)