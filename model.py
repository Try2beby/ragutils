import os
from typing import Any, Dict, List
from abc import abstractmethod
from llama_index.core.base.query_pipeline.query import (
    ChainableMixin,
)
from llama_index.core.bridge.pydantic import Field, validator
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import BaseComponent
from llama_index.core.base.llms.types import CompletionResponse

# from llama_index.core.llms.llm import LLM
# from llama_index.core.base.llms.base import BaseLLM


# class BaseLLM(ChainableMixin, BaseComponent):
class BaseLLM:
    """BaseLLM interface."""

    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )

    class Config:
        arbitrary_types_allowed = True

    @validator("callback_manager", pre=True)
    def _validate_callback_manager(cls, v: CallbackManager) -> CallbackManager:
        if v is None:
            return CallbackManager([])
        return v

    @property
    @abstractmethod
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        """Completion endpoint for LLM.

        If the LLM is a chat model, the prompt is transformed into a single `user` message.

        Args:
            prompt (str):
                Prompt to send to the LLM.
            formatted (bool, optional):
                Whether the prompt is already formatted for the LLM, by default False.
            kwargs (Any):
                Additional keyword arguments to pass to the LLM.

        Returns:
            CompletionResponse: Completion response from the LLM.

        Examples:
            ```python
            response = llm.complete("your prompt")
            print(response.text)
            ```
        """


# llamacpp, ollama, vllm, lmdeploy
class OpenAIAPILikeLLM(BaseLLM):
    def __init__(self, **config):
        super().__init__(**config)
        from openai import OpenAI

        if "api_key" in config:
            self.api_key = config["api_key"]
        else:
            self.api_key = os.environ.get("LLM_API_KEY", "Bearer no-key")

        self.model = self.config.get("model", "model")
        self.stop = self.config.get("stop", None)
        self.stream = self.config.get("stream", True)
        self.base_url = self.config.get("base_url", None)
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def generate(self, history: List[Dict[str, Any]], prompt: str = None, **kwargs):
        if prompt is None:
            prompt = ""
        chat_history = [{"role": "system", "content": prompt}] + history
        response = self.client.chat.completions.create(
            messages=chat_history,
            model=kwargs.get("model", self.model),
            stream=self.stream,
            max_tokens=kwargs.get("max_tokens", 128),
            temperature=kwargs.get("temperature", 0.2),
            stop=kwargs.get("stop", self.stop),
        )

        if not self.stream:
            yield response.choices[0].message.content, response
        else:
            for chunk in response:
                text = chunk.choices[0].delta.content
                if text is None:
                    text = ""
                yield text, chunk

    def complete(self, prompt: str, formatted: bool = False, **kwargs: Any):
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            formatted=formatted,
            **kwargs,
        )
        return response
