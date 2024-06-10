import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
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


class NvidiaRetrievalModel(DRESModel):
    # Refer to the code of DRESModel for the methods to overwrite
    def __init__(self, model_name, task_name: str):
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.prefix = get_task_def_by_task_name_and_type(task_name, "Retrieval")

        self.prompt = None
        self.gpu_count = torch.cuda.device_count()
        if self.gpu_count > 1:
            self.encoder = torch.nn.DataParallel(self.encoder)

        self.encoder.cuda()
        self.encoder.eval()

    def encode_queries(self, queries: List[str], **kwargs) -> np.ndarray:
        input_texts = ["Instruct: "+self.prefix+"\nQuery: " + q for q in queries]
        print(f"Prompt: {self.prefix}")
        return self._do_encode(input_texts)

    def encode_corpus(self, corpus: List[Dict[str, str]], **kwargs) -> np.ndarray:
        input_texts = ['{} {}'.format(doc.get('title', ''), doc['text']).strip() for doc in corpus]
        # doesn't need a prefix for docs
        return self._do_encode(input_texts)

    @torch.no_grad()
    def _do_encode(self, input_texts: List[str], prefix: str = "") -> np.ndarray:
        encoded_embeds = []
        batch_size = 12 * self.gpu_count
        for start_idx in tqdm.tqdm(range(0, len(input_texts), batch_size), desc='encoding', mininterval=10):
            batch_input_texts: List[str] = input_texts[start_idx: start_idx + batch_size]

            with torch.cuda.amp.autocast():
                outputs = self.encoder.encode(batch_input_texts, instruction=prefix, max_length=4096)
                embeds = F.normalize(outputs, p=2, dim=-1)
                encoded_embeds.append(embeds.cpu().numpy())

        return np.concatenate(encoded_embeds, axis=0)




