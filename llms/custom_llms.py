from typing import List, Mapping, Any, Sequence
import os

from llama_index.core.llms import (
    CustomLLM,
    ChatResponse,
    CompletionResponse,
    CompletionResponseGen,
    ChatMessage,
    ChatResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_chat_callback, llm_completion_callback

from llama_index.core.base.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)

from openai import OpenAI
from .api import api_generate


def messages_to_dict(messages: Sequence[ChatMessage]) -> List[Mapping[str, str]]:
    return [{"role": message.role, "content": message.content} for message in messages]


class Llama3(CustomLLM):
    context_window: int = 8192 - 512
    num_output: int = 512
    model_name: str = "Llama3"
    dummy_response: str = "My response"
    is_chat_model: bool = True
    api_path: str = "http://192.168.101.15:8081/v1/chat_completion/"

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=self.is_chat_model,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        completion_response = self.complete(messages, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        completion_response_gen = self.stream_complete(
            messages, formatted=True, **kwargs
        )
        return stream_completion_response_to_chat_response(completion_response_gen)

    @llm_completion_callback()
    def complete(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> CompletionResponse:
        messages_dict = messages_to_dict(messages)
        response = api_generate(
            api_path=self.api_path,
            messages=messages_dict,
            model="llama3",
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.2),
            verbose=True,
            stream=False,
        )
        text, _ = next(response)
        return CompletionResponse(text=text)

    @llm_completion_callback()
    def stream_complete(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> CompletionResponseGen:
        messages_dict = messages_to_dict(messages)
        response_gen = api_generate(
            api_path=self.api_path,
            messages=messages_dict,
            model="llama3",
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.2),
            verbose=True,
            stream=True,
        )
        completion_response_gen = (
            CompletionResponse(text=response[0]) for response in response_gen
        )
        return completion_response_gen


class DeepSeek(CustomLLM):
    context_window: int = 32768 - 1024
    num_output: int = 1024
    model_name: str = "deepseek-chat"
    # dummy_response: str = "My response"
    is_chat_model: bool = True

    client: Any

    def __init__(self):
        super().__init__()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
            is_chat_model=self.is_chat_model,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        completion_response = self.complete(messages, formatted=True, **kwargs)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        completion_response_gen = self.stream_complete(
            messages, formatted=True, **kwargs
        )
        return stream_completion_response_to_chat_response(completion_response_gen)

    @llm_completion_callback()
    def complete(
        self, messages: Sequence[ChatMessage], temperature: float = 0.7, **kwargs: Any
    ) -> CompletionResponse:
        messages_dict = messages_to_dict(messages)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages_dict,
            # messages=messages,
            stream=False,
            temperature=temperature,
        )
        return CompletionResponse(text=response.choices[0].message.content)

    @llm_completion_callback()
    def stream_complete(
        self, messages: Sequence[ChatMessage], temperature: float = 0.7, **kwargs: Any
    ) -> CompletionResponseGen:
        messages_dict = messages_to_dict(messages)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages_dict,
            stream=True,
            temperature=temperature,
        )
        yield from response


def get_custom_llm(llm_name: str) -> CustomLLM:
    if llm_name == "Llama3":
        return Llama3()
    elif llm_name == "DeepSeek":
        return DeepSeek()
    else:
        raise ValueError(f"Unknown LLM name: {llm_name}")
