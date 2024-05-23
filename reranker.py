import copy
import numpy as np
from typing import List, Tuple, Optional
from langchain_community.docstore.document import Document
from ranx import Run, fuse

from llmparty.utils.misc import make_md5
from llmparty.utils.triton_client_pack import TritonClient

from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import MetadataMode, NodeWithScore, QueryBundle

from llama_index.core.schema import (
    TextNode,
)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class BGEM3TritonReranker:
    def __init__(
        self,
        url: str = "localhost:9500",
        model_name: str = "reranker",
        client_type: str = "http",
        bach_size: int = 32,
        top_n: int = 2,
        keep_retrieval_score: Optional[bool] = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.client = TritonClient(
            url, model_name=model_name, client_type=client_type, **kwargs
        )
        self.bach_size = bach_size
        self.input_key = "input_pairs"
        self.output_key = "scores"

        self.top_n = top_n
        self.keep_retrieval_score = keep_retrieval_score

    def get_scores(self, query: str, texts: List[str]) -> List[float]:
        try:
            output = self.client.infer(
                input_dict={self.input_key: [[query, text] for text in texts]},
                output_names=[self.output_key],
            )
            scores = sigmoid(output[self.output_key].reshape(-1))
            return scores.tolist()
        except Exception as e:
            print(
                f"Exception occurred while trying to get scores: {str(e)}"
            )  # noqa: T201
            raise e

    def postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
        query_str: Optional[str] = None,
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        if query_str is not None and query_bundle is not None:
            raise ValueError("Cannot specify both query_str and query_bundle")
        elif query_str is not None:
            query_bundle = QueryBundle(query_str)
        else:
            pass
        return self._postprocess_nodes(nodes, query_bundle)

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if query_bundle is None:
            raise ValueError("Missing query bundle in extra info.")
        if len(nodes) == 0:
            return []

        scores = []
        for i in range(0, len(nodes), self.bach_size):
            batch_nodes = nodes[i : i + self.bach_size]
            texts = [
                node.node.get_content(metadata_mode=MetadataMode.EMBED)
                for node in batch_nodes
            ]
            scores.extend(self.get_scores(query_bundle.query_str, texts))

        assert len(scores) == len(nodes)

        for node, score in zip(nodes, scores):
            if self.keep_retrieval_score:
                # keep the retrieval score in metadata
                node.node.metadata["retrieval_score"] = node.score
            node.score = score

        new_nodes = sorted(nodes, key=lambda x: -x.score if x.score else 0)[
            : self.top_n
        ]

        return new_nodes


class FusedReranker:
    def __init__(self, method: str, norm: str = None, params: dict = None) -> None:
        self.method = method
        self.norm = norm
        self.params = params or {}

    def rerank_nodes_with_scores(
        self,
        results_list: List[List[Tuple[TextNode, float]]],
        k: int = 3,
        return_all_hybrid_results: bool = False,
    ) -> List[Tuple[TextNode, float]]:
        runs = [Run() for _ in range(len(results_list))]

        point_id2doc = {}
        for i in range(len(results_list)):
            results = results_list[i]
            for j in range(len(results)):
                point_id = make_md5(results[j][0].get_content())
                if point_id not in point_id2doc:
                    point_id2doc[point_id] = copy.deepcopy(results[j][0])
                runs[i].add_score("query", point_id, results[j][1])

        fused_scores = fuse(
            runs=runs, norm=self.norm, method=self.method, **self.params
        )

        results = []
        for point_id in sorted(
            fused_scores["query"], key=lambda x: fused_scores["query"][x], reverse=True
        ):
            score = fused_scores["query"][point_id]
            doc = point_id2doc[point_id]
            doc.metadata["score"] = score
            results.append([doc, score])

            if not return_all_hybrid_results and len(results) >= k:
                break
        return results

    def rerank_nodes(
        self,
        results_list: List[List[Tuple[TextNode, float]]],
        k: int = 3,
        return_all_hybrid_results: bool = False,
    ) -> List[TextNode]:
        results = self.rerank_nodes_with_scores(
            results_list, k, return_all_hybrid_results
        )
        return [result[0] for result in results]


def create_reranker(reranker_type: str = None, **kwargs):
    if reranker_type == "bgem3_reranker":
        return BGEM3TritonReranker(**kwargs.get("bgem3_reranker", {}))
    elif reranker_type == "fused_reranker":
        return FusedReranker(**kwargs.get("fused_reranker", {}))
    else:
        raise NotImplementedError(f"Unsupported reranker type: {reranker_type}")
