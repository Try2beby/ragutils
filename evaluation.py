from llama_index.core.evaluation import (
    EmbeddingQAFinetuneDataset,
    # RetrieverEvaluator,
)
from llama_index.core.evaluation.retrieval.base import BaseRetrievalEvaluator
from llama_index.core.evaluation.retrieval.evaluator import RetrieverEvaluator

from .globals import logger
from .utils import display_results
from ._metrics import resolve_metrics

from typing import Any, List


class MyRetrieverEvaluator(RetrieverEvaluator):
    @classmethod
    def from_metric_names(
        cls, metric_names: List[str], **kwargs: Any
    ) -> "BaseRetrievalEvaluator":
        """Create evaluator from metric names.

        Args:
            metric_names (List[str]): List of metric names
            **kwargs: Additional arguments for the evaluator

        """
        metric_types = resolve_metrics(metric_names)
        return cls(metrics=[metric() for metric in metric_types], **kwargs)


class Evaluator:
    def __init__(self, retriever, dataset: str | EmbeddingQAFinetuneDataset):
        self.retriever = retriever
        self.evaluator = self._get_evaluator(retriever)
        if isinstance(dataset, str):
            self.qa_dataset = EmbeddingQAFinetuneDataset.from_json(dataset)
        elif isinstance(dataset, EmbeddingQAFinetuneDataset):
            self.qa_dataset = dataset
        else:
            raise ValueError(
                "dataset must be a path or EmbeddingQAFinetuneDataset, but got: "
                f"{type(dataset)}"
            )

    def _get_evaluator(self, retriever):
        return MyRetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=retriever
        )

    async def aevaluate(self):
        # example
        sample_id, sample_query = list(self.qa_dataset.queries.items())[101]
        sample_expected = self.qa_dataset.relevant_docs[sample_id]

        eval_result = self.evaluator.evaluate(sample_query, sample_expected)
        logger.info(eval_result)

        eval_results = await self.evaluator.aevaluate_dataset(
            self.qa_dataset,
            show_progress=True,
            # workers=4
        )

        return display_results("", eval_results)
