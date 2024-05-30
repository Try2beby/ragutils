import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
import chromadb.utils.embedding_functions as embedding_functions

# from llama_index.vector_stores.qdrant import QdrantVectorStore
from .mys.vector_store import MyQdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient

from llama_index.core.schema import BaseNode

from enum import Enum
import os
from typing import Optional, Sequence, Dict

from .globals import logger

# from .sparse import load_sparse_model, make_sparse_vectors_func

from llmparty.demo.rag_based_chat.embedding import BGEM3TritonEmbeddings as Embeddings
from .embeding.dense import MultilingualEmbeddings


class USING_DB(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"


def get_vector_index(
    nodes: Optional[Sequence[BaseNode]] = None,
    usingdb: USING_DB = USING_DB.QDRANT,
    embedding: Embeddings = Embeddings(),
    vectorstore_config: Dict = None,
):
    logger.info("Building/Getting Vector Index")
    logger.info(f"Using {usingdb}")

    use_multilingual = vectorstore_config.get("use_multilingual", False)
    logger.info(f"Using Multilingual: {use_multilingual}")
    if use_multilingual:
        dense_embedding = MultilingualEmbeddings()
    else:
        dense_embedding = embedding

    if usingdb == USING_DB.CHROMA:
        # initialize client, setting path to save data
        db = chromadb.PersistentClient(path="./chroma")
        huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
            model_name="BAAI/bge-m3", api_key=os.getenv("HUGGINGFACE_API_KEY")
        )

        # create collection
        chroma_collection = db.get_or_create_collection(
            "squad_sub",
            metadata={"hnsw:space": "cosine"},
            embedding_function=huggingface_ef,
        )

        # assign chroma as the vector_store to the context
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if nodes is not None:
            vector_index = VectorStoreIndex(
                nodes,
                storage_context=storage_context,
                embed_model="local:BAAI/bge-m3",
                show_progress=True,
            )
        else:
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store, embed_model="local:BAAI/bge-m3"
            )

        logger.info(f"#{vector_index._vector_store._collection.count()} nodes indexed.")

    elif usingdb == USING_DB.QDRANT:
        qdrant_kwargs: Dict = vectorstore_config.get("qdrant_kwargs", {})

        url = qdrant_kwargs.pop("url", "http://localhost:6333")
        # creates a persistant index to disk
        client = QdrantClient(url=url)
        aclient = AsyncQdrantClient(url=url)

        # create our vector store with hybrid indexing enabled
        vector_store = MyQdrantVectorStore(
            client=client,
            aclient=aclient,
            batch_size=32,
            sparse_doc_fn=embedding.as_sparse_doc_fn(),
            sparse_query_fn=embedding.as_sparse_query_fn(),
            **qdrant_kwargs,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if nodes is not None:
            vector_index = VectorStoreIndex(
                nodes,
                insert_batch_size=128,
                storage_context=storage_context,
                # embed_model="local:BAAI/bge-m3",
                # embed_model=embedding,
                embed_model=dense_embedding,
                show_progress=True,
            )
        else:
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=dense_embedding,
                show_progress=True,
            )

    else:
        raise NotImplementedError(f"Invalid value for usingdb, got: {usingdb}")
        # raise ValueError(
        #     "Invalid value for usingdb, must be either 'chroma' or 'qdrant', got: {usingdb}"
        # )

    return vector_index
