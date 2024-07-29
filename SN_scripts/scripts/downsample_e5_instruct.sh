models=(
    "intfloat/e5-mistral-7b-instruct"
)

declare -A model_dims=(
    [intfloat/e5-mistral-7b-instruct]=4096
)

for model in "${models[@]}"; do
    # skip the first one
    skipped_first=False
    while IFS=, read -r dataset_name split lang subsplit
    do
        if [ $skipped_first = False ]; then
            skipped_first=True
            continue
        fi
        echo "$dataset_name $lang $split $subsplit"
        /home/toolkit/./eai job new -f SN_scripts/config/default.yaml --field id -- /bin/bash -c \
        "source /opt/conda/bin/activate /home/toolkit/mteb-lite/.conda && \
        bash run_all.sh $dataset_name $model ${model_dims[$model]} $split $subsplit \
        >> /home/toolkit/mteb-lite/$dataset_name-${models[@]//\//-}-$split-$subsplit.log 2>&1"
    done < tasks_to_downsample.csv
done
