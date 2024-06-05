from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import BaseNode

from typing import Dict, Optional

from .utils import json_to_doc


class MySimpleDocumentStore(SimpleDocumentStore):
    @property
    def docs(self) -> Dict[str, BaseNode]:
        """Get all documents.

        Returns:
            Dict[str, BaseDocument]: documents

        """
        json_dict = self._kvstore.get_all(collection=self._node_collection)
        return {key: json_to_doc(json) for key, json in json_dict.items()}

    def get_document(self, doc_id: str, raise_error: bool = True) -> Optional[BaseNode]:
        """Get a document from the store.

        Args:
            doc_id (str): document id
            raise_error (bool): raise error if doc_id not found

        """
        json = self._kvstore.get(doc_id, collection=self._node_collection)
        if json is None:
            if raise_error:
                raise ValueError(f"doc_id {doc_id} not found.")
            else:
                return None
        return json_to_doc(json)
