from llama_index.core.schema import (
    TextNode,
)
from llama_index.core.bridge.pydantic import Field
from typing import Any, Dict
from typing_extensions import Self
from enum import Enum, auto


class ObjectType(str, Enum):
    TEXT = auto()
    IMAGE = auto()
    INDEX = auto()
    DOCUMENT = auto()
    TEXT_CTX = auto()


class MyTextNode(TextNode):
    context_texts: list[str] = Field(default=[], description="Context texts.")
    context_ids: list[str] = Field(default=[], description="Context IDs.")

    def __init__(self, context_texts, context_ids, **kwargs):
        super().__init__(**kwargs)
        self.context_texts = context_texts
        self.context_ids = context_ids

    @classmethod
    def class_name(cls) -> str:
        return "TextCtxNode"

    @classmethod
    def get_type(cls) -> str:
        """Get Object type."""
        return ObjectType.TEXT_CTX

    # TODO: return type here not supported by current mypy version
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> Self:  # type: ignore
        if isinstance(kwargs, dict):
            data.update(kwargs)

        data.pop("class_name", None)
        return cls(**data)
