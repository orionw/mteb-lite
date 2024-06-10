import argparse
import os
from functools import partial

from mteb import MTEB
import torch

from models.downloaded_gritlm import GritLM


DTYPE_TO_TORCH_DTYPE = {
    'bfloat16': torch.bfloat16,
    'float32': torch.float32,
    'float16': torch.float16,
}

def get_gpus_max_memory(max_memory):
    max_memory = {i: max_memory for i in range(torch.cuda.device_count())}
    return max_memory

def gritlm_instruction_format(instruction):
    return "<|user|>\n" + instruction + "\n<|embed|>\n" if instruction else "<|embed|>\n"

def zephyr_instruction_format(instruction):
    return "<|user|>\n" + instruction + "</s>\n<|assistant|>\n"

def tulu_instruction_format(instruction):
    return "<|user|>\n" + instruction + "\n<|assistant|>\n"

def mistral_instruction_format(instruction):
    return "[INST] " + instruction + " [/INST] "

NAME_TO_FUNC = {
    "gritlm": gritlm_instruction_format,
    "zephyr": zephyr_instruction_format,
    "tulu": tulu_instruction_format,
    "mistral": mistral_instruction_format,
}

SET_TO_FEWSHOT_PROMPT = {
    "e5": {
        "Retrieval": '\n\nFor example given "{}", you should retrieve "{}"',
        "Other": '\n\nFor example given "{}", it would match with "{}"',
    },
    "medi2": {
        "Retrieval": '\n\nThe provided query could be "{}" and the positive "{}"',
        "Other": '\n\nThe provided query could be "{}" and the positive "{}"',
    },
}


# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model_name_or_path', default="GritLM/GritLM-7B", type=str)
#     parser.add_argument('--attn_implementation', default='sdpa', type=str, help="eager/sdpa/flash_attention_2")
#     parser.add_argument('--attn', default='bbcc', type=str, help="only first two letters matter for embedding")
#     parser.add_argument('--task_types', default=None, help="Comma separated. Default is None i.e. running all tasks")
#     parser.add_argument('--task_names', default=None, help="Comma separated. Default is None i.e. running all tasks")
#     parser.add_argument('--instruction_set', default="e5", type=str, help="Instructions to use")
#     parser.add_argument('--instruction_format', default="gritlm", type=str, help="Formatting to use")
#     parser.add_argument('--no_instruction', action='store_true', help="Do not use instructions")
#     parser.add_argument('--batch_size', default=32, type=int)
#     parser.add_argument('--max_length', default=None, type=int)
#     parser.add_argument('--num_shots', default=None, type=int)
#     parser.add_argument('--dtype', default='bfloat16', type=str)
#     parser.add_argument('--output_dir', default=None, type=str)
#     parser.add_argument('--overwrite_results', action='store_true')
#     parser.add_argument('--pipeline_parallel', action='store_true')
#     parser.add_argument('--embedding_head', default=None, type=str)
#     parser.add_argument('--pooling_method', default='mean', type=str)
#     parser.add_argument('--save_qrels', action='store_true')
#     parser.add_argument('--top_k', default=1000, type=int)    
#     return parser.parse_args()

def get_gritlm_model(model_name_or_path: str):
    # save args defaults
    attn_implementation = "sdpa"
    attn = "bbcc"
    instruction_set = "e5"
    instruction_format = "gritlm"
    no_instruction = False
    batch_size = 12
    max_length = None
    num_shots = None
    dtype = "bfloat16"
    output_dir = None
    overwrite_results = False
    pipeline_parallel = False
    embedding_head = None
    pooling_method = "mean"
    save_qrels = False
    top_k = 1000

    # don't allow instruction_set to be anything but the default or no_instruction to be true
    if instruction_set != "e5" or no_instruction:
        raise ValueError("Instruction will be given from the dataset, don't try to pass arguments")


    # Quick skip if exists
    model_name = model_name_or_path.rstrip('/').split('/')[-1]

    kwargs = {
        "model_name_or_path": model_name_or_path,
        # Normalizing embeddings will harm the performance of classification task
        # as it does not use the cosine similarity
        # For other tasks, cosine similarity is used in the evaluation, 
        # so embeddings are automatically normalized
        "normalized": False,
        "torch_dtype": DTYPE_TO_TORCH_DTYPE.get(dtype, torch.bfloat16),
        "mode": "embedding",
        "pooling_method": pooling_method,
        "attn_implementation": attn_implementation,
        "attn": attn,
    }

    if pipeline_parallel:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = get_gpus_max_memory("50GB")
        kwargs["offload_folder"] = "offload"

    if any([x in model_name_or_path for x in ["instructor"]]):
        assert kwargs["pooling_method"] == "mean"
    elif any([x in model_name_or_path for x in ["bge"]]):
        assert kwargs["pooling_method"] == "cls"
    
    if pooling_method == "lasttoken":
        kwargs["embed_eos"] = "</e>"
    if embedding_head:
        kwargs["projection"] = embedding_head

    model = GritLM(**kwargs)
    if embedding_head:
        model.load_state_dict(
            torch.load(model_name_or_path + "/embedding_head.bin"), strict=False,
        )
        model.projection.to(model.device)

    if os.getenv("BIDIRECTIONAL_ATTN", False):
        model.model.padding_idx = model.tokenizer.pad_token_id
        if hasattr(model.model, "model"):
            model.model.model.padding_idx = model.tokenizer.pad_token_id
        if hasattr(model.model, "module"):
            model.model.module.padding_idx = model.tokenizer.pad_token_id            
    
    if max_length is not None:
        model.encode = partial(model.encode, max_length=max_length)

    return model