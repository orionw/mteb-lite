#!/bin/bash

shard_num=$1
shard_total=$2
dataset=$3
model=$4
dim_size=$5
split=$6
subsplit=$7
normalize=$8


# if normalize is empty, make it true
if [ -z "$normalize" ]; then
  normalize=true
fi

# if split is empty, use `test`
if [ -z "$split" ]; then
  split=test
fi

# if subsplit is non-empty, set the flag
if [ ! -z "$subsplit" ]; then
  subsplit="--subsplit $subsplit"
fi

file_safe_dataset=$(echo $dataset | sed 's/\//--/g')
file_safe_model=$(echo $model | sed 's/\//_/g')

echo "Dataset: $dataset"
echo "Model: $model"
echo "Dim size: $dim_size"
echo "Split: $split"
echo "Path to file in mteb: $path_to_file_in_mteb"
echo "Normalize: $normalize"
echo "File safe dataset: $file_safe_dataset"
echo "File safe model: $file_safe_model"
echo "Subsplit: $subsplit"

# check if it exists otherwise run it (embedding_{args.shard}--{args.shard_total}.jsonl')
if [ ! -f "indexes/${file_safe_dataset}-$split/$file_safe_model/embedding_${shard_num}--${shard_total}.jsonl" ]; then
    echo "Embedding the corpus..."
    python embed_corpus_shards.py --dataset $dataset --model $model --split $split --shard $shard_num --shard_total $shard_total $subsplit 
fi
