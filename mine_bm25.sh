#!/bin/bash

# example usage: bash ./mine_bm25.sh NFCorpus en test

dataset_name=$1
lang=$2
split=$3
subsplit=$4

# if subsplit is non-empty add it
if [ ! -z "$subsplit" ]; then
  subsplit="--subsplit $subsplit"
fi

echo "Downloading $dataset_name..."
python download_all.py \
  --dataset_name $dataset_name \
  --split $split $subsplit
  

echo "Indexing $dataset_name..."
python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input datasets/${dataset_name}--${split}/corpus/ \
  --language $lang \
  --index indexes/${dataset_name}--${split} \
  --generator DefaultLuceneDocumentGenerator \
  --threads 12 \
  --storePositions

echo "Searching $dataset_name with BM25..."
python -m pyserini.search.lucene \
  --index indexes/${dataset_name}--${split} \
  --topics datasets/${dataset_name}--${split}/queries.tsv \
  --output datasets/${dataset_name}--${split}/run_${dataset_name}--${split}.trec \
  --language $lang \
  --bm25