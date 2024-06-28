declare -A tasks_dict=(
    [MIRACLRetrieval]="en de fr es ru ja zh fa ar fi ko id th te hi"
    [TopiOCQA]="validation"
    [MSMARCO-PL]="test"
    [MSMARCO]="test"
    [FEVER]="test"
    [ClimateFEVER]="test"
    [HotpotQA]="test"
    [HotpotQA-PL]="test"
    [DBPedia]="test"
    [DBPedia-PL]="test"
    [NeuCLIR2022Retrieval]="rus zho fas"
    [NeuCLIR2023Retrieval]="rus zho fas"
    [NQ]="test"
    [NQ-PL]="test"
    [RiaNewsRetrieval]="test"
    [QuoraRetrieval]="test"
    [Quora-PL]="test validation"
)

tasks=("${!tasks_dict[@]}")

models=(
    "intfloat/multilingual-e5-large"
)

declare -A model_dims=(
    [intfloat/multilingual-e5-large]=1024
)

for model in "${models[@]}"; do
    for task in "${tasks[@]}"; do
        for split in ${tasks_dict[$task]}; do
            echo "Running $task $model $split"
            /home/toolkit/./eai job new -f SN_scripts/config/default.yaml --field id -- /bin/bash -c \
            "source /opt/conda/bin/activate /home/toolkit/mteb-lite/.conda && \
            bash run_all.sh $task $model ${model_dims[$model]} $split \
            >> /home/toolkit/mteb-lite/$task-${models[@]//\//-}-$split.log 2>&1"
        done
    done
done
