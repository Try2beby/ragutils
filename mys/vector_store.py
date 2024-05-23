from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.types import VectorStoreQueryResult
from llama_index.core.vector_stores.utils import (
    legacy_metadata_dict_to_node,
    # metadata_dict_to_node,
)


from .utils import metadata_dict_to_node

from qdrant_client.http.models import Payload

from .node import MyTextNode, TextNode

from typing import Any, List, cast

from qdrant_client.http.models import (
    Payload,
)

DENSE_VECTOR_NAME = "text-dense"
SPARSE_VECTOR_NAME_OLD = "text-sparse"
SPARSE_VECTOR_NAME = "text-sparse-new"


class MyQdrantVectorStore(QdrantVectorStore):
    def parse_to_query_result(self, response: List[Any]) -> VectorStoreQueryResult:
        """
        Convert vector store response to VectorStoreQueryResult.

        Args:
            response: List[Any]: List of results returned from the vector store.
        """
        nodes = []
        similarities = []
        ids = []

        for point in response:
            payload = cast(Payload, point.payload)
            try:
                node = metadata_dict_to_node(payload)
            except Exception:
                metadata, node_info, relationships = legacy_metadata_dict_to_node(
                    payload
                )

                node = TextNode(
                    id_=str(point.id),
                    text=payload.get("text"),
                    metadata=metadata,
                    start_char_idx=node_info.get("start", None),
                    end_char_idx=node_info.get("end", None),
                    relationships=relationships,
                )
            nodes.append(node)
            similarities.append(point.score)
            ids.append(str(point.id))

        return VectorStoreQueryResult(nodes=nodes, similarities=similarities, ids=ids)
