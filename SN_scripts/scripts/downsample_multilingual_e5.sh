models=(
    "intfloat/multilingual-e5-large"
)

declare -A model_dims=(
    [intfloat/multilingual-e5-large]=1024
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
        bash run_all.sh $task $model ${model_dims[$model]} $split $subsplit \
        >> /home/toolkit/mteb-lite/$task-${models[@]//\//-}-$split-$subsplit.log 2>&1"
    done < tasks_to_downsample.csv
done
