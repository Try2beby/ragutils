from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.docstore.types import BaseDocumentStore
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from .mys.retriever import (
    SimpleHybridRetriever,
    RetrieverWithReRank,
    MyQueryFusionRetriever,
    # TruncateRetriever,
)
from typing import Dict, Optional

from .globals import IS_JUPYTER, CONSOLE, get_device, logger
from .utils import display_source_node_cmd, display_source_node

import sys

sys.path.append("..")


def get_vector_retriever(
    vector_index: VectorStoreIndex,
    vector_retriever_kwargs: Dict = {},
):
    vector_retriever = vector_index.as_retriever(**vector_retriever_kwargs)

    return vector_retriever


def get_sparse_retriever(
    docstore: Optional[BaseDocumentStore],
):
    sparse_retriever = BM25Retriever.from_defaults(
        docstore=docstore, similarity_top_k=20
    )
    return sparse_retriever


def get_hybrid_retriever(vector_retriever, sparse_retriever):
    return SimpleHybridRetriever(
        vector_retriever=vector_retriever, sparse_retriever=sparse_retriever
    )


def get_retriever_with_rerank(
    retriever, rerank_model: str = "BAAI/bge-reranker-v2-m3", top_n: int = 20
):
    reranker = SentenceTransformerRerank(
        top_n=top_n, model=rerank_model, device=str(get_device())
    )
    return RetrieverWithReRank(retriever=retriever, reranker=reranker)


def get_retriever_with_rerank_triton(retriever, reranker_config):
    from .reranker import create_reranker

    reranker = create_reranker(**reranker_config)
    return RetrieverWithReRank(retriever=retriever, reranker=reranker)


def get_query_fusion_retriever(
    retrievers=[],
    top_n: int = 5,
    num_queries: int = 4,
    similarity_top_k: int = 20,
    query_gen_prompt: str = None,
):
    return MyQueryFusionRetriever(
        retrievers=retrievers,
        similarity_top_k=similarity_top_k,
        top_n=top_n,
        num_queries=num_queries,  # set this to 1 to disable query generation
        mode="reciprocal_rerank",
        use_async=True,
        verbose=False,
        query_gen_prompt=query_gen_prompt,
    )


def retriever_test(query: str, retriever: BaseRetriever, source_length: int = 600):
    # print name of retriever

    retrieved_nodes = retriever.retrieve(query)
    logger.info(f"Retrieved {len(retrieved_nodes)} nodes for query: {query}")

    # from llama_index.core.response.notebook_utils import display_source_node

    if IS_JUPYTER:
        for node in retrieved_nodes[:3]:
            display_source_node(node, source_length=source_length)
    else:
        CONSOLE.rule()
        for node in retrieved_nodes[:3]:
            display_source_node_cmd(node, source_length=source_length)
        CONSOLE.rule()
