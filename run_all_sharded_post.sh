#!/bin/bash

shard_total=$1
dataset=$2
model=$3
dim_size=$4
split=$5
subsplit=$6
normalize=$7

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
echo "Normalize: $normalize"
echo "File safe dataset: $file_safe_dataset"
echo "File safe model: $file_safe_model"
echo "Subsplit: $subsplit"


# check if the faiss index exists, otherwise convert
if [ ! -f "indexes/${file_safe_dataset}-$split/$file_safe_model/full/index" ]; then
    echo "Converting to faiss..."
    bash convert_to_faiss_shards.sh indexes/${file_safe_dataset}-$split/$file_safe_model/ $dim_size $shard_total
fi

# check if the queries file exists in artifacts, otherwise download
if [ ! -f "artifacts/${file_safe_dataset}-$split.tsv" ]; then
    echo "Downloading queries..."
    python download_queries.py --dataset $dataset --split $split $subsplit
fi

# check if the run file exists in artifacts, otherwise run
if [ ! -f "artifacts/run_${file_safe_model}_${file_safe_dataset}-$split.tsv" ]; then
    # Mine the run file with the hard negatives per corpus embedding using `bash mine_hard_negatives.sh INDEX_FOLDER/full QUERIES_FILE NORMALIZE_TRUE_OR_FALSE BATCH_SIZE NUM_HITS MODEL_NAME`
    echo "Mining hard negatives..."
    bash mine_hard_negatives.sh indexes/${file_safe_dataset}-$split/$file_safe_model/full artifacts/${file_safe_dataset}-$split.tsv $normalize 32 1000 $model $subsplit
fi
