from typing import Any, List, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from .globals import get_device


def load_sparse_model():

    extra_kwargs = {}
    extra_kwargs["model"] = {}
    extra_kwargs["model"]["torch_dtype"] = torch.float16
    extra_kwargs["model"]["low_cpu_mem_usage"] = True
    extra_kwargs["model"]["device_map"] = {"": get_device()}

    doc_tokenizer = AutoTokenizer.from_pretrained(
        "naver/efficient-splade-VI-BT-large-doc"
    )
    doc_model = AutoModelForMaskedLM.from_pretrained(
        "naver/efficient-splade-VI-BT-large-doc", **extra_kwargs["model"]
    )

    query_tokenizer = AutoTokenizer.from_pretrained(
        "naver/efficient-splade-VI-BT-large-query",
    )
    query_model = AutoModelForMaskedLM.from_pretrained(
        "naver/efficient-splade-VI-BT-large-query", **extra_kwargs["model"]
    )

    return {
        "doc_tokenizer": doc_tokenizer,
        "doc_model": doc_model,
        "query_tokenizer": query_tokenizer,
        "query_model": query_model,
    }


def make_sparse_vectors_func(model, tokenizer):
    def sparse_vectors(
        texts: List[str],
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Computes vectors from logits and attention mask using ReLU, log, and max operations.
        """
        tokens = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
        tokens = tokens.to(get_device())

        output = model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        tvecs, _ = torch.max(weighted_log, dim=1)

        # extract the vectors that are non-zero and their indices
        indices = []
        vecs = []
        for batch in tvecs:
            indices.append(batch.nonzero(as_tuple=True)[0].tolist())
            vecs.append(batch[indices[-1]].tolist())

        # # back to cpu
        # indices = [idxs.cpu().numpy().tolist() for idxs in indices]
        # vecs = [v.cpu().numpy().tolist() for v in vecs]

        return indices, vecs

    return sparse_vectors
