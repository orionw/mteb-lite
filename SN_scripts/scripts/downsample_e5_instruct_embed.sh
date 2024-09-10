models=(
    "intfloat/e5-mistral-7b-instruct"
)

declare -A model_dims=(
    [intfloat/e5-mistral-7b-instruct]=4096
)

declare -A num_shards_map=(
    [MIRACLRetrieval-th]=8
    [MIRACLRetrieval-en]=32
    [MIRACLRetrieval-de]=32
    [MIRACLRetrieval-fr]=32
    [MIRACLRetrieval-es]=16
    [MIRACLRetrieval-ru]=16
    [MIRACLRetrieval-ja]=8
    [MIRACLRetrieval-fa]=8
    [MIRACLRetrieval-ar]=8
    [MIRACLRetrieval-fi]=8
    [MIRACLRetrieval-ko]=8
    [MIRACLRetrieval-id]=8
    [MIRACLRetrieval-te]=8
    [MIRACLRetrieval-hi]=8
    [MIRACLRetrieval-zh]=8
    [FEVER-test]=8
    [ClimateFEVER-test]=8
    [HotpotQA-test]=8
    [DBPedia-test]=8
    [NQ-test]=8
    [NeuCLIR2023Retrieval-rus]=8
    [NeuCLIR2023Retrieval-zho]=8
    [NeuCLIR2023Retrieval-fas]=8
    [NeuCLIR2022Retrieval-rus]=8
    [NeuCLIR2022Retrieval-zho]=8
    [NeuCLIR2022Retrieval-fas]=8
    [RiaNewsRetrieval-test]=8
    [QuoraRetrieval-test]=8
    [TopiOCQA-validation]=32
    [MSMARCO-test]=8
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
        num_shards=${num_shards_map[$dataset_name-$split]}
        for i in $(seq 0 $((num_shards-1))); do
            if [ -f "indexes/$dataset_name-$split/${models[@]//\//_}/embedding_$i--$num_shards.jsonl" ]; then
                continue
            fi
            echo "$dataset_name $lang $split $subsplit $i $num_shards"
            eai job new -f SN_scripts/config/default.yaml --field id -- /bin/bash -c \
            "source /opt/conda/bin/activate /home/toolkit/mteb-lite/.conda && \
            bash run_all_sharded_embed.sh $i $num_shards $dataset_name $model ${model_dims[$model]} $split $subsplit \
            >> /home/toolkit/mteb-lite/$dataset_name-${models[@]//\//-}-$split-$subsplit-$i-$num_shards.log 2>&1"
        done
    done < tasks_to_downsample.csv
done

