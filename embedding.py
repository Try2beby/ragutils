from llmparty.demo.rag_based_chat.embedding import create_embeddings

from typing import Dict


def get_embedding_model(embedding_config: Dict = None):
    return create_embeddings(**embedding_config)
