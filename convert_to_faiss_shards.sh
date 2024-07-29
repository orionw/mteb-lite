#!/bin/bash


base_folder=$1
dim_size=$2
shards_total=$3

cd $base_folder
echo "$base_folder"

echo "Converting to faiss"
for (( i=0; i<$shards_total; i++ ))
do
    shard_folder=$(printf "%02d" $i)
    mkdir -p $shard_folder
    mkdir -p $i

    shard_file="embedding_${i}--${shards_total}.jsonl"
    if [ ! -f "$shard_file" ]; then
        echo "File $shard_file does not exist, skipping"
    else
        mv $shard_file $shard_folder/embedding.jsonl
    fi

    # if the index file exists skip it
    if [ -f "${i}/index" ]; then
        echo "Index file exists, skipping"
    else
        echo "python -m pyserini.index.faiss --dim $dim_size --input ${shard_folder}/ --output ${i}/"
        python -m pyserini.index.faiss --dim $dim_size --input "${shard_folder}/" --output "${i}/"
    fi
done


echo "python -m pyserini.index.merge_faiss_indexes --prefix ./ --shard-num $shards_total --dim $dim_size"
python -m pyserini.index.merge_faiss_indexes --prefix ./ --shard-num $shards_total --dim $dim_size


# # Clean up and remove all the shard folders
for (( i=0; i<$shards_total; i++ ))
do
    shard_folder=$(printf "%02d" $i)
    rm -rf $shard_folder
    rm -rf $i
done



cd --

