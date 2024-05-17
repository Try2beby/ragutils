from typing import Any, Dict, Optional

from llama_index.core.constants import DATA_KEY, TYPE_KEY
from llama_index.core.storage.docstore.utils import legacy_json_to_doc
from llama_index.core.schema import (
    BaseNode,
    Document,
    ImageDocument,
    ImageNode,
    IndexNode,
    TextNode,
)

from .node import MyTextNode


def metadata_dict_to_node(metadata: dict, text: Optional[str] = None) -> BaseNode:
    """Common logic for loading Node data from metadata dict."""
    node_json = metadata.get("_node_content", None)
    node_type = metadata.get("_node_type", None)
    if node_json is None:
        raise ValueError("Node content not found in metadata dict.")

    node: BaseNode
    if node_type == IndexNode.class_name():
        node = IndexNode.from_json(node_json)
    elif node_type == ImageNode.class_name():
        node = ImageNode.from_json(node_json)
    elif node_type == MyTextNode.class_name():
        node = MyTextNode.from_json(node_json)
    else:
        node = TextNode.from_json(node_json)

    if text is not None:
        node.set_content(text)

    return node


def json_to_doc(doc_dict: dict) -> BaseNode:
    doc_type = doc_dict[TYPE_KEY]
    data_dict = doc_dict[DATA_KEY]
    doc: BaseNode

    if "extra_info" in data_dict:
        return legacy_json_to_doc(doc_dict)
    else:
        if doc_type == Document.get_type():
            doc = Document.from_dict(data_dict)
        elif doc_type == ImageDocument.get_type():
            doc = ImageDocument.from_dict(data_dict)
        elif doc_type == TextNode.get_type():
            doc = TextNode.from_dict(data_dict)
        elif doc_type == MyTextNode.get_type():
            doc = MyTextNode.from_dict(data_dict)
        elif doc_type == ImageNode.get_type():
            doc = ImageNode.from_dict(data_dict)
        elif doc_type == IndexNode.get_type():
            doc = IndexNode.from_dict(data_dict)
        else:
            raise ValueError(f"Unknown doc type: {doc_type}")

        return doc
