from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.core.evaluation import RetrieverEvaluator

from .globals import QA_DATASET, logger
from .print_utils import display_results


class Evaluator:
    def __init__(self, retriever, dataset=QA_DATASET):
        self.retriever = retriever
        self.evaluator = self.get_evaluator(retriever)
        self.qa_dataset = EmbeddingQAFinetuneDataset.from_json(dataset)

    def get_evaluator(self, retriever):
        return RetrieverEvaluator.from_metric_names(
            ["mrr", "hit_rate"], retriever=retriever
        )

    async def aevaluate(self):
        # example
        sample_id, sample_query = list(self.qa_dataset.queries.items())[101]
        sample_expected = self.qa_dataset.relevant_docs[sample_id]

        eval_result = self.evaluator.evaluate(sample_query, sample_expected)
        logger.info(eval_result)

        eval_results = await self.evaluator.aevaluate_dataset(
            self.qa_dataset, show_progress=True, workers=4
        )

        return display_results("", eval_results)
