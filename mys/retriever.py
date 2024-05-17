# 导入整个模块
from typing import Coroutine, List
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.indices.vector_store.retrievers import VectorIndexRetriever

# from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from llama_index.core.vector_stores.types import VectorStoreQuery

from typing import Any


class SimpleHybridRetriever(BaseRetriever):
    def __init__(
        self, vector_retriever: VectorIndexRetriever, sparse_retriever: BaseRetriever
    ):
        self.vector_retriever = vector_retriever
        self.sparse_retriever = sparse_retriever
        super().__init__()

    def _retrieve(self, query_bundle, **kwargs):
        sparse_nodes = self.sparse_retriever.retrieve(query_bundle, **kwargs)
        vector_nodes = self.vector_retriever.retrieve(query_bundle, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in sparse_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes

    async def _aretrieve(self, query_bundle, **kwargs):
        # print("using _aretrieve")
        bm25_nodes = await self.bm25_retriever.aretrieve(query_bundle, **kwargs)
        vector_nodes = await self.vector_retriever.aretrieve(query_bundle, **kwargs)

        # combine the two lists of nodes
        all_nodes = []
        node_ids = set()
        for n in bm25_nodes + vector_nodes:
            if n.node.node_id not in node_ids:
                all_nodes.append(n)
                node_ids.add(n.node.node_id)
        return all_nodes


class RetrieverWithReRank(BaseRetriever):
    def __init__(self, retriever, reranker):
        self.retriever = retriever
        self.reranker = reranker

    def _retrieve(self, query_bundle, **kwargs):
        # print("using _retrieve with reranker")
        retrieved_nodes = self.retriever.retrieve(query_bundle, **kwargs)
        reranked_nodes = self.reranker.postprocess_nodes(
            retrieved_nodes,
            query_bundle=query_bundle,
        )
        return reranked_nodes

    async def _aretrieve(self, query_bundle, **kwargs):
        # print("using _aretrieve with reranker")
        retrieved_nodes = await self.retriever.aretrieve(query_bundle, **kwargs)
        reranked_nodes = self.reranker.postprocess_nodes(
            retrieved_nodes,
            query_bundle=query_bundle,
        )
        return reranked_nodes

    def retrieve(
        self,
        str_or_query_bundle: str | QueryBundle,
        top_n: int = 20,
    ) -> List[NodeWithScore]:
        return super().retrieve(str_or_query_bundle)[:top_n]

    def aretrieve(
        self,
        str_or_query_bundle: str | QueryBundle,
        top_n: int = 20,
    ) -> Coroutine[Any, Any, List[NodeWithScore]]:
        return super().aretrieve(str_or_query_bundle)[:top_n]


class MyQueryFusionRetriever(QueryFusionRetriever):
    def __init__(self, top_n: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.top_n = top_n

    def _retrieve(self, query_bundle, **kwargs):
        retrieved_nodes = super()._retrieve(query_bundle, **kwargs)
        return retrieved_nodes[: self.top_n]

    async def _aretrieve(self, query_bundle, **kwargs):
        retrieved_nodes = await super()._aretrieve(query_bundle, **kwargs)
        return retrieved_nodes[: self.top_n]


# class TruncateRetriever(BaseRetriever):
#     def __init__(self, retriever: BaseRetriever, top_n: int = 5):
#         self.retriever = retriever
#         self.top_n = top_n

#     def _retrieve(self, query_bundle, **kwargs):
#         return self.retriever.retrieve(query_bundle, **kwargs)[: self.top_n]

#     async def _aretrieve(self, query_bundle, **kwargs):
#         retrieved_nodes = await self.retriever.aretrieve(query_bundle, **kwargs)
#         return retrieved_nodes[: self.top_n]
