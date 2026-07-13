import threading

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline as hf_pipeline,
)

MODEL_ID = "cross-encoder/nli-distilroberta-base"

_pipeline = None
_lock = threading.Lock()


def get_nli_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:
            return _pipeline
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
        if torch.cuda.is_available():
            model = model.cuda()
        model.eval()
        _pipeline = hf_pipeline(
            "zero-shot-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            multi_label=True,
        )
        return _pipeline
