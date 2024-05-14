import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
import chromadb.utils.embedding_functions as embedding_functions

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient

from llama_index.core.schema import BaseNode

from enum import Enum
import os
from typing import Optional, Sequence, Dict

from .globals import logger
from .sparse import load_sparse_model, make_sparse_vectors_func


class USING_DB(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"


def get_vector_index(
    nodes: Optional[Sequence[BaseNode]] = None,
    usingdb: USING_DB = USING_DB.QDRANT,
    qdrant_kwargs: Dict = {},
):
    logger.info("Building/Getting Vector Index")
    logger.info(f"Using {usingdb}")

    # openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    #                 api_key=os.getenv('OPENAI_API_KEY'),
    #                 api_base=os.getenv('OPENAI_API_BASE'),
    #             )

    huggingface_ef = embedding_functions.HuggingFaceEmbeddingFunction(
        model_name="BAAI/bge-m3", api_key=os.getenv("HUGGINGFACE_API_KEY")
    )

    if usingdb == USING_DB.CHROMA:
        # initialize client, setting path to save data
        db = chromadb.PersistentClient(path="./chroma")

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

        # print(f"#{vector_index._vector_store._collection.count()} nodes indexed.")
        logger.info(f"#{vector_index._vector_store._collection.count()} nodes indexed.")

    elif usingdb == USING_DB.QDRANT:
        # creates a persistant index to disk
        client = QdrantClient(host="localhost", port=6333)
        aclient = AsyncQdrantClient(host="localhost", port=6333)

        model_tokenizer_dict = load_sparse_model()

        # create our vector store with hybrid indexing enabled
        vector_store = QdrantVectorStore(
            "squad_sub",
            client=client,
            aclient=aclient,
            enable_hybrid=True,
            batch_size=32,
            sparse_doc_fn=make_sparse_vectors_func(
                model=model_tokenizer_dict["doc_model"],
                tokenizer=model_tokenizer_dict["doc_tokenizer"],
            ),
            sparse_query_fn=make_sparse_vectors_func(
                model=model_tokenizer_dict["query_model"],
                tokenizer=model_tokenizer_dict["query_tokenizer"],
            ),
            **qdrant_kwargs,
        )

        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        if nodes is not None:
            vector_index = VectorStoreIndex(
                nodes,
                insert_batch_size=128,
                storage_context=storage_context,
                embed_model="local:BAAI/bge-m3",
                show_progress=True,
            )
        else:
            vector_index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model="local:BAAI/bge-m3",
                show_progress=True,
            )

    else:
        # raise a error
        raise ValueError("Invalid value for 'usingdb'")

    return vector_index
