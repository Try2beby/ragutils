from llama_index.core.evaluation.retrieval.metrics_base import (
    BaseRetrievalMetric,
)
from llama_index.core.evaluation.retrieval.metrics import (
    HitRate,
    MRR,
    CohereRerankRelevancyMetric,
)

from typing import List, Type, Dict


METRIC_REGISTRY: Dict[str, Type[BaseRetrievalMetric]] = {
    "hit_rate": HitRate,
    "mrr": MRR,
    "cohere_rerank_relevancy": CohereRerankRelevancyMetric,
}


def resolve_metrics(metrics: List[str]) -> List[Type[BaseRetrievalMetric]]:
    """Resolve metrics from list of metric names."""
    for metric in metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Invalid metric name: {metric}")

    return [METRIC_REGISTRY[metric] for metric in metrics]
