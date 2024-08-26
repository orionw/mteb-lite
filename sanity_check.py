import subprocess
import os

DATASET_DIR = "/home/toolkit/mteb-lite/artifacts"

ARTIFACT_DIR = "/home/toolkit/mteb-lite-run-files-e5/artifacts"

DATASETS = [
    "ClimateFEVER-test",
    "DBPedia-test",
    "FEVER-test",
    "HotpotQA-test",
    "MIRACLRetrieval-ar",
    "MIRACLRetrieval-de",
    "MIRACLRetrieval-es",
    "MIRACLRetrieval-fa",
    "MIRACLRetrieval-fi",
    "MIRACLRetrieval-fr",
    "MIRACLRetrieval-hi",
    "MIRACLRetrieval-id",
    "MIRACLRetrieval-ja",
    "MIRACLRetrieval-ko",
    "MIRACLRetrieval-ru",
    "MIRACLRetrieval-te",
    "MIRACLRetrieval-th",
    "MIRACLRetrieval-zh",
    "MSMARCO-test",
    "NeuCLIR2022Retrieval-fas",
    "NeuCLIR2022Retrieval-rus",
    "NeuCLIR2022Retrieval-zho",
    "NeuCLIR2023Retrieval-fas",
    "NeuCLIR2023Retrieval-rus",
    "NeuCLIR2023Retrieval-zho",
    "NQ-test",
    "QuoraRetrieval-test",
    "RiaNewsRetrieval-test",
]

MODELS = [
    "intfloat_e5-mistral-7b-instruct",
    "intfloat_multilingual-e5-large",
]

def get_num_lines_of_file(file_path):
    # do with bash command
    result = subprocess.run(
        ["wc", "-l", file_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return int(result.stdout.decode().split()[0])

if __name__ == "__main__":
    for dataset in DATASETS:
        dataset_file = f"{DATASET_DIR}/{dataset}.tsv"
        if not os.path.isfile(dataset_file):
            print(f"Dataset {dataset_file} does not exist")
            continue
        num_lines_dataset = get_num_lines_of_file(dataset_file)

        for model in MODELS:
            artifact_file = f"{ARTIFACT_DIR}/run_{model}_{dataset}.tsv"
            if not os.path.isfile(artifact_file):
                print(f"Artifact {artifact_file} does not exist")
                continue
            num_lines_artifact = get_num_lines_of_file(artifact_file)

            if num_lines_dataset * 1000 != num_lines_artifact:
                print(f"Dataset {dataset} and Artifact {artifact_file} have different number of lines")