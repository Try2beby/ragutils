import os
from typing import Any, Awaitable, Callable, List, Optional, Sequence

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseAsyncGen,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseAsyncGen,
    CompletionResponseGen,
    MessageRole,
)


def completion_response_to_chat_response(
    completion_response: CompletionResponse,
) -> ChatResponse:
    """Convert a completion response to a chat response."""
    return ChatResponse(
        message=ChatMessage(
            role=MessageRole.ASSISTANT,
            content=completion_response.text,
            additional_kwargs=completion_response.additional_kwargs,
        ),
        raw=completion_response.raw,
    )


def stream_completion_response_to_chat_response(
    completion_response_gen: CompletionResponseGen,
) -> ChatResponseGen:
    """Convert a stream completion response to a stream chat response."""

    def gen() -> ChatResponseGen:
        for response in completion_response_gen:
            yield ChatResponse(
                message=ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.text,
                    additional_kwargs=response.additional_kwargs,
                ),
                delta=response.delta,
                raw=response.raw,
            )

    return gen()
