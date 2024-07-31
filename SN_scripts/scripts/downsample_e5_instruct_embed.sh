models=(
    "intfloat/e5-mistral-7b-instruct"
)

declare -A model_dims=(
    [intfloat/e5-mistral-7b-instruct]=4096
)

num_shards=8

for model in "${models[@]}"; do
    # skip the first one
    skipped_first=False
    while IFS=, read -r dataset_name split lang subsplit
    do
        if [ $skipped_first = False ]; then
            skipped_first=True
            continue
        fi
        for i in $(seq 0 $((num_shards-1))); do
            echo "$dataset_name $lang $split $subsplit $i $num_shards"
            eai job new -f SN_scripts/config/default.yaml --field id -- /tk/bin/start.sh bin/bash -c \
            "source /opt/conda/bin/activate /home/toolkit/mteb-lite/.conda && \
            bash run_all_sharded_embed.sh $i $num_shards $dataset_name $model ${model_dims[$model]} $split $subsplit \
            >> /home/toolkit/mteb-lite/$dataset_name-${models[@]//\//-}-$split-$subsplit-$i-$num_shards.log 2>&1"
        done
    done < tasks_to_downsample.csv
done

