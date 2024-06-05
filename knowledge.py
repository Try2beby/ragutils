from .data import Data, QADataSet
from .retriever import (
    get_sparse_retriever,
    get_vector_retriever,
    get_retriever_with_rerank_triton,
    get_query_fusion_retriever,
    retriever_test,
)
from .metrics.beir import MyBeirEvaluator

from .index import get_vector_index, VectorStoreIndex
from .embeding.embedding import get_embedding_model
from .globals import logger

import sys
import os

sys.path.append("../")
from llmparty.utils.config import load_yaml
from llmparty.demo.rag_based_chat.embedding import BGEM3TritonEmbeddings as Embeddings


from typing import Dict
from enum import Enum


class RETRIEVER_SCHEME(str, Enum):
    SPARSE = "sparse"
    VECTOR = "vector"
    HYBRID = "hybrid"
    QUERY_FUSION = "query_fusion"


class Knowledge:
    def __init__(
        self,
        chunk_config: Dict,
        reranker_config: Dict,
        vectorstore_config: Dict,
        embedding_config: Dict,
    ):
        self.data = Data(**chunk_config)
        self.paths = self.resolve_paths()
        self.nodes = self.data.get_nodes()

        self.dataset = self.resolve_dataset(path=self.paths["qa_dataset"])
        self.embedding = self.resolve_embedding(embedding_config=embedding_config)
        self.vector_index = self.resolve_vector_index(
            vectorstore_config=vectorstore_config, embedding=self.embedding
        )
        self.retriever = self.resolve_retriever(
            vector_index=self.vector_index,
            scheme=RETRIEVER_SCHEME.QUERY_FUSION,
            use_reranker=True,
            reranker_config=reranker_config,
        )

    def resolve_paths(self):
        chunk_size = self.data.chunk_options["chunk_size"]
        chunk_overlap = self.data.chunk_options["chunk_overlap"]
        context_window = self.data.context_windows
        store = os.path.join(
            "./store", f"docstore_{chunk_size}_{chunk_overlap}_{context_window}.json"
        )
        qa_dataset = os.path.join(
            "./data", f"qa_dataset_{chunk_size}_{chunk_overlap}_{context_window}.json"
        )
        paths = dict(
            store=store,
            qa_dataset=qa_dataset,
        )

        return paths

    def evaluation(self):
        query_test = "Apa itu SLIK OJK?"
        retriever_test(query=query_test, retriever=self.retriever)
        eval_results = MyBeirEvaluator().run(
            retriever=self.retriever,
            dataset="./data/qa_dataset.json",
            metrics_k_values=[1, 3, 5, 10, 20],
        )
        return eval_results

    def resolve_embedding(self, embedding_config: Dict = None):
        return get_embedding_model(embedding_config)

    def resolve_retriever(
        self,
        vector_index: VectorStoreIndex = None,
        scheme: RETRIEVER_SCHEME = RETRIEVER_SCHEME.QUERY_FUSION,
        use_reranker: bool = True,
        reranker_config: Dict = None,
    ):
        retriever = None

        if scheme == RETRIEVER_SCHEME.QUERY_FUSION:
            vector_retriever = get_vector_retriever(
                vector_index=vector_index,
                vector_retriever_kwargs={
                    "vector_store_query_mode": "default",
                },
            )

            sparse_retriever = get_vector_retriever(
                vector_index=vector_index,
                vector_retriever_kwargs={
                    "vector_store_query_mode": "sparse",
                },
            )

            fusion_retriever_kwargs = dict(
                query_gen_prompt=(
                    "You are an expert in legal knowledge. Generate {num_queries} search queries, one on each line, that may be helpful in answering the following input query. Limit your response to Indonesian language:\n"
                    "Query: {query}\n"
                    "Queries:\n"
                ),
            )
            retriever = get_query_fusion_retriever(
                [vector_retriever, sparse_retriever],
                num_queries=4,
                top_n=20,
                **fusion_retriever_kwargs,
            )

        elif scheme == RETRIEVER_SCHEME.SPARSE:
            retriever = get_vector_retriever(
                vector_index=self.vector_index,
                vector_retriever_kwargs={
                    "vector_store_query_mode": "sparse",
                },
            )

        elif scheme == RETRIEVER_SCHEME.VECTOR:
            retriever = get_vector_retriever(
                vector_index=self.vector_index,
                vector_retriever_kwargs={
                    "vector_store_query_mode": "default",
                },
            )

        elif scheme == RETRIEVER_SCHEME.HYBRID:
            retriever = get_vector_retriever(
                vector_index=self.vector_index,
                vector_retriever_kwargs={
                    "vector_store_query_mode": "hybrid",
                },
            )

        if use_reranker:
            return get_retriever_with_rerank_triton(
                retriever, reranker_config=reranker_config
            )
        else:
            return retriever

    def resolve_vector_index(
        self,
        create: bool = False,
        vectorstore_config: Dict = None,
        embedding: Embeddings = Embeddings(),
    ):
        usingdb = vectorstore_config.pop("usingdb", None)
        if create:
            vector_index = get_vector_index(
                nodes=self.nodes,
                usingdb=usingdb,
                embedding=embedding,
                vectorstore_config=vectorstore_config,
            )
        else:
            vector_index = get_vector_index(
                usingdb=usingdb,
                embedding=embedding,
                vectorstore_config=vectorstore_config,
            )

        return vector_index

    def resolve_dataset(self, path: str = None):
        try:
            qa_dataset = QADataSet.from_persist_path(path)
        except FileNotFoundError:
            qa_dataset = QADataSet.from_nodes(
                self.nodes,
                retriever=get_sparse_retriever(docstore=self.data.get_docstore()),
            )

        return qa_dataset

    @classmethod
    def from_yaml(
        cls, path: str, overrides: Dict = None, if_print: bool = False
    ) -> "Knowledge":
        """Load knowledge from YAML file."""
        config = load_yaml(path, overrides)

        chunk_config = config.get("chunk", None)
        reranker_config = config.get("reranker", None)
        vectorstore_config = config.get("vectorstore", None)
        embedding_config = config.get("embedding", None)

        if if_print:
            logger.info("Chunk config: ", chunk_config)
            logger.info("Reranker config: ", reranker_config)
            logger.info("Vectorstore config: ", vectorstore_config)
            logger.info("Embedding config: ", embedding_config)

        return cls(
            chunk_config=chunk_config,
            reranker_config=reranker_config,
            vectorstore_config=vectorstore_config,
            embedding_config=embedding_config,
        )
