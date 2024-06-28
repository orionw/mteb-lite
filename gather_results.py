from tqdm import tqdm
import argparse
import json
import os
import os
import pandas as pd
import numpy as np
import mteb
import Stemmer

import bm25s.hf


def index_bm25(args, dataset):
    # We will use the snowball stemmer from the PyStemmer library and tokenize the corpus
    stemmer = Stemmer.Stemmer("english")
    dataset_list = [(doc.get("title", "") + doc["text"]).strip() for doc in dataset]
    corpus_tokenized = bm25s.tokenize(dataset_list, stemmer=stemmer)

    # We create a BM25 retriever, index the corpus, and save to Hugging Face Hub
    retriever = bm25s.hf.BM25HF()
    retriever.index(corpus_tokenized)

    retriever.save_to_hub(repo_id=args.repo_path, token=os.getenv("HF_TOKEN"), corpus=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed a dataset")
    parser.add_argument("--model", type=str, help="The model name", default="intfloat/e5-mistral-7b-instruct")
    parser.add_argument("--dataset", type=str, help="The dataset name", default="NQ")
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size")
    parser.add_argument("--split", type=str, default="test", help="The split to embed")
    parser.add_argument("--repo_path", type=str, required=True, help="Where to upload the index")
    args = parser.parse_args()

    # Load the dataset
    tasks = mteb.get_tasks(tasks=[args.dataset])
    assert len(tasks) == 1, f"Expected 1 task, got {len(tasks)}"
    tasks[0].load_data()
    corpus = tasks[0].corpus[args.split]
    index_bm25(args, corpus)