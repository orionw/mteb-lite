#!/bin/bash


original_file=$1
dim_size=$2
base_folder=$(dirname $original_file)

cd $base_folder
echo "$base_folder"
base_name_only=$(basename $original_file)
# split it into 10 pieces for easier processing
new_name_prefix=$(echo $base_name_only | sed 's/\.jsonl//g')
echo "split -n l/10 -d $base_name_only $new_name_prefix"
split -n l/10 -d $base_name_only $new_name_prefix

echo "Converting to faiss"
for i in {00..09}
do
    mkdir -p $i
    mkdir -p ${i: -1}
    if [ ! -f "$new_name_prefix$i" ]; then
        echo "File $new_name_prefix$i does not exist, skipping"
    else
        mv $new_name_prefix$i $i/$new_name_prefix.jsonl
    fi

    # if the index file exists skip it
    if [ -f "${i: -1}/index" ]; then
        echo "Index file exists, skipping"
    else
        echo "python -m pyserini.index.faiss --dim $dim  --input ${i}/ --output ${i: -1}/"
        python -m pyserini.index.faiss --dim $dim_size --input "${i}/" --output ${i: -1}/
    fi

done


echo "python -m pyserini.index.merge_faiss_indexes --prefix ./ --shard-num 10 --dim $dim_size"
python -m pyserini.index.merge_faiss_indexes --prefix ./ --shard-num 10 --dim $dim_size

## clean up and remove all the double digits
for i in {00..09}
do
    rm -rf $i
done
# also remove the single digits now that we have the full
for i in {0..9}
do
    rm -rf $i
done
# now remove the big file
rm $original_file

cd --

