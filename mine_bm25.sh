#!/bin/bash

dataset_folder=$1
lang=$2
dataset_name=$(echo $dataset_folder | basename)

echo "Indexing $dataset_name..."
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $dataset_folder \
  --language $lang \
  --index indexes/$dataset_name \
  --generator DefaultLuceneDocumentGenerator \
  --threads 12 \
  --storePositions

echo "Searching $dataset_name with BM25..."
python -m pyserini.search.lucene \
  --index indexes/$dataset_name \
  --topics $queries \
  --output results/run_${dataset_name}.trec \
  --language $lang \
  --bm25