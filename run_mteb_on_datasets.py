import mteb
import os
import json
import argparse
from sentence_transformers import SentenceTransformer
from models.e5_models import E5RetrievalModel
from models.gritlm_models import get_gritlm_model
from models.nvidia_models import NvidiaRetrievalModel


DOCS_VALUES = [2, 5, 10, 50, 100, 500]


SENTENCE_TRANSFORMER_MODELS = [
    "facebook/contriever-msmarco",
    "BAAI/bge-large-en-v1.5",
    "Alibaba-NLP/gte-base-en-v1.5",
]

E5_MODELS = [
    "intfloat/e5-small-v2",
    "intfloat/e5-large-v2",
    "intfloat/multilingual-e5-large",
    "intfloat/e5-mistral-7b-instruct",
    "Salesforce/SFR-Embedding-Mistral"
]

GRITLM_MODELS = [
    "GritLM/GritLM-7B"
]

NVIDIA_MODELS = [
    "nvidia/NV-Embed-v1"
]

def load_model(model_name: str, task_name: str = None):
    if model_name in SENTENCE_TRANSFORMER_MODELS:
        return SentenceTransformer(model_name, trust_remote_code=True)
    elif model_name in E5_MODELS:
        return E5RetrievalModel(model_name, task_name)
    elif model_name in GRITLM_MODELS:
        return get_gritlm_model(model_name, task_name)
    elif model_name in NVIDIA_MODELS:
        return NvidiaRetrievalModel(model_name, task_name)
    else:
        raise ValueError(f"Model {model_name} not found")


def run_all_mteb(args):
    tasks = [args.dataset_name.split("/")[-1].replace('-', "_") + f"_top_{num}_only" for num in DOCS_VALUES]
    tasks += [args.dataset_name.split("/")[-1].replace('-', "_") + f"_top_{num}_only_w_correct" for num in DOCS_VALUES]
    # now add the real task
    tasks += ["NQ"] # to get a comparison in case I ran these wrong
    for model in SENTENCE_TRANSFORMER_MODELS + E5_MODELS + NVIDIA_MODELS:
        print(f"Running model {model}")
        evaluation = mteb.MTEB(tasks=tasks)
        loaded_model = load_model(model, args.dataset_name.split("/")[-1])
        evaluation.run(loaded_model, eval_splits=["test"], output_folder="results/" + model.replace("/", "--"))

    print("Finished running all tasks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all MTEB tasks")
    parser.add_argument("--dataset_name", type=str, help="The dataset name", required=True)
    args = parser.parse_args()

    run_all_mteb(args)

    # example usage:
    #   python run_mteb_on_datasets.py --dataset_name mteb/nq
    #  python run_mteb_on_datasets.py --dataset_name mteb/trec-covid