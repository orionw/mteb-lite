#!/bin/bash

# example run:
#   bash run_all.sh mteb/trec-covid intfloat/e5-large-v2 1024 mteb_repo/mteb/tasks/Retrieval/eng/TRECCOVIDRetrieval.py

dataset=$1
model=$2
dim_size=$3
split=$4
subsplit=$5
normalize=$6
path_to_file_in_mteb=$7

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

# check if indexes/$file_safe_dataset/$file_safe_model/embedding.jsonl exists, otherwise embed
if [ ! -f "indexes/${file_safe_dataset}-$split/$file_safe_model/embedding.jsonl" ]; then
    echo "Embedding the corpus..."
    python embed_corpus.py --dataset $dataset --model $model --split $split $subsplit
fi

# check if indexes/$file_safe_dataset/$file_safe_model/embedding.jsonl exists, otherwise thow an error
if [ ! -f "indexes/${file_safe_dataset}-$split/$file_safe_model/embedding.jsonl" ]; then
    echo "Embedding failed. Exiting..."
    exit 1
fi


# check if the faiss index exists, otherwise convert
if [ ! -f "indexes/${file_safe_dataset}-$split/$file_safe_model/full/index" ]; then
    echo "Converting to faiss..."
    bash convert_to_faiss.sh indexes/${file_safe_dataset}-$split/$file_safe_model/embedding.jsonl $dim_size
fi

# check if the faiss index exists, otherwise thow an error
if [ ! -f "indexes/${file_safe_dataset}-$split/$file_safe_model/full/index" ]; then
    echo "Converting to faiss failed. Exiting..."
    exit 1
fi

# check if the queries file exists in artifacts, otherwise download
if [ ! -f "artifacts/${file_safe_dataset}-$split.tsv" ]; then
    echo "Downloading queries..."
    python download_queries.py --dataset $dataset --split $split $subsplit
fi

# check if the queries file exists in artifacts, otherwise thow an error
if [ ! -f "artifacts/${file_safe_dataset}-$split.tsv" ]; then
    echo "Downloading queries failed. Exiting..."
    exit 1
fi

# check if the run file exists in artifacts, otherwise run
if [ ! -f "artifacts/run_${file_safe_model}_${file_safe_dataset}-$split.tsv" ]; then
    # Mine the run file with the hard negatives per corpus embedding using `bash mine_hard_negatives.sh INDEX_FOLDER/full QUERIES_FILE NORMALIZE_TRUE_OR_FALSE BATCH_SIZE NUM_HITS MODEL_NAME`
    echo "Mining hard negatives..."
    bash mine_hard_negatives.sh indexes/${file_safe_dataset}-$split/$file_safe_model/full artifacts/${file_safe_dataset}-$split.tsv $normalize 32 1000 $model $subsplit
fi

# if the dataset isn't on HF, add it. This you'll have to comment out manually if it was already done (From here on out)
# echo "Adding to HF..."
# python subselect_and_create_datasets.py --run_files artifacts/run_$file_safe_dataset.tsv --dataset_name $dataset


# # run and add to the mteb repo if it isn't there already
# echo "Adding to mteb..."
# python automatically_add_dataset_to_mteb.py --original_repo_name $dataset --path_to_existing $path_to_file_in_mteb

# # # Run the results on all, it will skip if already ran
# echo "Running on all datasets..."
# python run_mteb_on_datasets.py --dataset_name $dataset
