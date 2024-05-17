from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.schema import BaseNode

from typing import Dict

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
