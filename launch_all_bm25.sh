#!/bin/bash

# read in the `tasks_to_downsample.csv` file as csv
# for each row in the csv, run the following commands
# bash mine_bm25.sh $dataset_name $lang $split following the three inputs in the csv

# example usage: bash ./launch_all_bm25.sh 

# skip the first one
skipped_first=False

while IFS=, read -r dataset_name split lang subsplit
do
    if [ $skipped_first = False ]; then
        skipped_first=True
        continue
    fi

    # if the run file already exists, skip
    if [ -f "datasets/${dataset_name}--${split}/run_${dataset_name}--${split}.trec" ]; then
        echo "Run file already exists, skipping..."
        continue
    fi
    echo "$dataset_name $lang $split $subsplit"
    bash mine_bm25.sh $dataset_name $lang $split $subsplit
done < tasks_to_downsample.csv
