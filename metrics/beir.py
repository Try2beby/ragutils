from llama_index.core.evaluation.benchmarks.beir import BeirEvaluator
from llama_index.core.base.base_retriever import BaseRetriever
from beir.retrieval.evaluation import EvaluateRetrieval

import tqdm
import pandas as pd
from typing import List, Dict, Tuple

import json


def results_to_scores(
    results: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int],
    metrics: List[str] = ["hit_rate", "mrr"],
):
    scores = {}
    for query_id in results.keys():
        scores[query_id] = {}
        ranked_list = sorted(
            results[query_id].items(), key=lambda x: x[1], reverse=True
        )
        relevant_docs = qrels.get(query_id, {})
        relevant_docs_set = set(relevant_docs.keys())

        for k in k_values:
            hit_rate = 0.0
            mrr = 0.0
            found_relevant = False

            for i, (doc_id, _) in enumerate(ranked_list[:k]):
                if doc_id in relevant_docs_set:
                    hit_rate = 1.0
                    if not found_relevant:
                        mrr = 1.0 / (i + 1)
                        found_relevant = True

            if "hit_rate" in metrics:
                scores[query_id][f"hit_rate_{k}"] = hit_rate
            if "mrr" in metrics:
                scores[query_id][f"mrr_{k}"] = mrr

    return scores


class MyEvaluateRetrieval(EvaluateRetrieval):
    @staticmethod
    def evaluate(
        qrels: Dict[str, Dict[str, int]],
        results: Dict[str, Dict[str, float]],
        k_values: List[int],
        ignore_identical_ids: bool = True,
    ) -> Tuple[
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
        Dict[str, float],
    ]:

        ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
            qrels, results, k_values, ignore_identical_ids
        )

        # compute hit rate and mrr
        hit_rate = {}
        mrr = {}

        for k in k_values:
            hit_rate[f"Hit@{k}"] = 0.0
            mrr[f"MRR@{k}"] = 0.0

        scores = results_to_scores(
            results, qrels, k_values, metrics=["hit_rate", "mrr"]
        )
        for query_id in scores.keys():
            for k in k_values:
                hit_rate[f"Hit@{k}"] += scores[query_id][f"hit_rate_{k}"]
                mrr[f"MRR@{k}"] += scores[query_id][f"mrr_{k}"]

        for k in k_values:
            hit_rate[f"Hit@{k}"] = round(hit_rate[f"Hit@{k}"] / len(scores), 5)
            mrr[f"MRR@{k}"] = round(mrr[f"MRR@{k}"] / len(scores), 5)

        return ndcg, _map, recall, precision, hit_rate, mrr


class MyBeirEvaluator(BeirEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def _load_dataset(self, dataset_path: str) -> None:
        with open(dataset_path) as f:
            dataset = json.load(f)
        queries, corpus, relevant_docs = (
            dataset["queries"],
            dataset["corpus"],
            dataset["relevant_docs"],
        )
        qrels = {
            query_id: {doc_id: 1 for doc_id in relevant_docs[query_id]}
            for query_id in relevant_docs
        }

        return corpus, queries, qrels

    def run(
        self,
        retriever: BaseRetriever,
        dataset: str,
        metrics_k_values: List[int] = [3, 10],
    ) -> None:

        corpus, queries, qrels = self._load_dataset(dataset)

        results = {}
        for key, query in tqdm.tqdm(queries.items()):
            nodes_with_score = retriever.retrieve(query)
            results[key] = {
                node.node.id_: float(node.score) for node in nodes_with_score
            }

        ndcg, map_, recall, precision, hit_rate, mrr = MyEvaluateRetrieval.evaluate(
            qrels, results, metrics_k_values
        )

        # Create DataFrame
        data = {
            "NDCG": ndcg.values(),
            "MAP": map_.values(),
            "Recall": recall.values(),
            "Precision": precision.values(),
            "Hit Rate": hit_rate.values(),
            "MRR": mrr.values(),
        }

        df = pd.DataFrame(data).T
        df = df.rename_axis("Metrics", axis="index")
        df.columns = [f"@{k}" for k in metrics_k_values]

        print("Results for:", dataset)
        print(df)
        print("-------------------------------------")

        return df
