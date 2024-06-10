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
            batch_emb = model.encode(sentences, batch_size=args.batch_size)
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

    ### now save it to faiss to we can load it for search
    dir_name_of_file = os.path.dirname(file_name)
    # rm any existing docid and index file, so we can actually convert to faiss
    if os.path.isfile(os.path.join(dir_name_of_file, "docid")):
        os.remove(os.path.join(dir_name_of_file, "docid"))

    if os.path.isfile(os.path.join(dir_name_of_file, "index")):
        os.remove(os.path.join(dir_name_of_file, "index"))

    # print(f"Running `python -m pyserini.index.faiss --dim {dim_size} --input {dir_name_of_file} --output {dir_name_of_file}`")
    # os.system(f"python -m pyserini.index.faiss --dim {dim_size} --input {dir_name_of_file} --output {dir_name_of_file}")


def combine_doc(doc: dict):
    if "title" in doc:
        return doc["title"] + " " + doc["text"]
    else:
        return doc["text"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed a dataset")
    parser.add_argument("--model", type=str, help="The model name", default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument("--dataset", type=str, help="The dataset name", default="mteb/nq")
    parser.add_argument("--batch_size", type=int, default=1000000000, help="The batch size")
    args = parser.parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset, "corpus")["corpus"]
    input = [combine_doc(item) for item in dataset]
    ids = [item["id"] if "id" in item else item["_id"] for item in dataset]

    # Load the model
    if "mistral" in args.model:
        use_vllm = True
        model = LLM(model=args.model, enforce_eager=True, dtype="auto", disable_sliding_window=True)
    else:
        use_vllm = False
        model = SentenceTransformer(args.model)

    dataloader = DataLoader(Dataset.from_dict({"text": input, "id": ids}), batch_size=args.batch_size, num_workers=0)

    # Embed the dataset
    embed_dataset(args, dataloader, model, use_vllm=use_vllm)