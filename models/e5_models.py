import os
import json
import tqdm
import numpy as np
import torch
import argparse
import torch.nn.functional as F

from typing import List, Dict
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput
from mteb import MTEB, AbsTaskRetrieval
from mteb.evaluation.evaluators.RetrievalEvaluator import DRESModel

from models.utils import pool, logger, move_to_cuda, get_detailed_instruct, get_task_def_by_task_name_and_type, create_batch_dict, MODEL_NAME_TO_POOL_TYPE, MODEL_NAME_TO_PREFIX_TYPE


class E5RetrievalModel(DRESModel):
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, model_name, task_name: str, doc_as_query: bool = False):
        self.model_name = model_name
        self.prefix_type = MODEL_NAME_TO_PREFIX_TYPE[self.model_name.split("/")[-1]]
        self.pool_type = MODEL_NAME_TO_POOL_TYPE[self.model_name.split("/")[-1]]
        self.doc_as_query = doc_as_query

        self.prompt = get_task_def_by_task_name_and_type(task_name, "Retrieval")

        self.encoder = AutoModel.from_pretrained(self.model_name, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        if self.prefix_type == 'query_or_passage':
            input_texts = [f'query: {q}' for q in queries]
        else:
            print(f"Prompt: {self.prompt}")
            input_texts = [self.prompt + q for q in queries]

        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        if self.doc_as_query:
            return self.encode_queries([d['text'] for d in corpus], **kwargs)

        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        # no need to add prefix for instruct models
        if self.prefix_type == 'query_or_passage':
            input_texts = ['passage: {}'.format(t) for t in input_texts]

        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str]) -> np.ndarray:
        encoded_embeds = []
        batch_size = 12 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            batch_dict = create_batch_dict(self.tokenizer, batch_input_texts)
            batch_dict = move_to_cuda(batch_dict)

            with torch.cuda.amp.autocast():
                outputs: BaseModelOutput = self.encoder(**batch_dict)
                embeds = pool(outputs.last_hidden_state, batch_dict['attention_mask'], self.pool_type)
                embeds = F.normalize(embeds, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)

    def set_prompt(self, prompt: str):
        self.prompt = prompt