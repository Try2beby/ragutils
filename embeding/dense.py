from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

from typing import List, Optional

from ..globals import get_device


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


# 'intfloat/multilingual-e5-large-instruct'
class MultilingualEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large-instruct",
        batch_size: int = 4,
    ):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=get_device())

        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> Optional[List[List[float]]]:
        final_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            final_embeddings.extend(
                self.model.encode(texts[i : i + self.batch_size]).tolist()
            )
        return final_embeddings

    def embed_query(self, text: str) -> Optional[List[float]]:
        task = (
            "Given a web search query, retrieve relevant passages that answer the query"
        )
        return self.model.encode(get_detailed_instruct(task, text)).tolist()
