#!/bin/bash


# bash mine_hard_negatives.sh indexes/mteb--nq/intfloat_e5-large-v2/full mteb--nq.tsv true 64 1000 intfloat/e5-large-v2

# where to find the saved embeddings index
index_folder=$1

# path to the query file, or dataset name if default queries
query_file=$2

# retrieval options
l2_norm=$3
batch_size=$4
hits=$5

# get model from index file name, e.g. /intfloat_e5-large-v2/ -> /intfloat_e5-large-v2/ -> /intfloat/e5-large-v2/
model=$6

echo "Index folder: $index_folder"
echo "Query file: $query_file"
echo "L2 norm: $l2_norm"
echo "Model: $model"


# the file safe dataset name is the 2nd to last dir in the index folder
file_safe_clean_name=$(basename $(dirname $index_folder))
echo "File safe name is $file_safe_clean_name"

if [ ! -f "$query_file" ]; then
  echo "Query file does not exist, downloading..."
  # if the file exists at `/home/orionweller/retrieval-w-instructions/nfs/cached_datasets/{dataset}`
  if [ ! -f "/home/orionweller/retrieval-w-instructions/nfs/cached_datasets/$query_file/queries.tsv" ]; then
    python ./bin/download_dataset.py --dataset $query_file --type queries
  fi
  query_file=/home/orionweller/retrieval-w-instructions/nfs/cached_datasets/$file_safe_clean_name/queries.tsv
  query_cache=" --encoder-class auto --encoder $model"
else
  echo "Query file exists, using it."
  query_cache=" --encoder-class auto --encoder $model"
fi


# if hits is empty, set it to 1000
if [ -z "$hits" ]; then
  hits=10000
fi

# if batch_size is empty, set it to 64
if [ -z "$batch_size" ]; then
  batch_size=64
fi

# add --l2-norm if we want to use it
if [ "$l2_norm" = "true" ]; then
  l2_norm="--l2-norm"
else
  l2_norm=""
fi

# add "run_" to the beginning of the basename
model_file_safe_name=$(echo $model | sed 's/\//_/g')
output_file=$(dirname $query_file)/run_${model_file_safe_name}_$(basename $query_file)


echo "Output file: $output_file"
echo "Batch size: $batch_size"
echo "Hits: $hits"


cmd=`cat <<EOF
python -m pyserini.search.faiss \
  --threads 16 \
  --batch-size $batch_size \
  --index $index_folder \
  --topics $query_file \
  --output $output_file \
  --device cuda:0 \
  --hits $hits $l2_norm $query_cache
EOF
`
echo $cmd
eval $cmd


## Example Usage
#   Example 1: using the default queries for a dataset, but they haven't been downloaded
#     bash bin/search_index.sh nfs/indexes/beir--scifact--test/intfloat_e5-large-v2/ "beir/scifact/test" true 64 1000
#   Example 2: using a downloaded set of queries
#     bash bin/search_index.sh nfs/indexes/beir--scifact--test/intfloat_e5-large-v2/ "nfs/cached_datasets/scifact/queries.tsv" true 64 1000