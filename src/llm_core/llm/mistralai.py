# -*- coding: utf-8 -*-
from mistralai import Mistral
from dataclasses import dataclass
from typing import Callable

from .base import LLMBase
from ..settings import MISTRAL_API_KEY


def load_mistralai_client(llm, **kwargs):
    client = Mistral(api_key=MISTRAL_API_KEY, **kwargs)
    return client


def create_mistralai_completion(
    llm,
    model,
    messages,
    temperature,
    tools=None,
    tool_choice=None,
    schema=None,
):
    # As per Mistral Documentation to force usage of tool.
    if tool_choice:
        tool_choice = "any"

    completion = llm._client.chat.complete(
        model=model, messages=messages, tools=tools, tool_choice=tool_choice
    )
    return completion.dict()


@dataclass
class MistralAIModel(LLMBase):
    create_completion: Callable = create_mistralai_completion
    loader: Callable = load_mistralai_client
    loader_kwargs: dict = None

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_model()

    def load_model(self):
        kwargs = self.loader_kwargs or {}
        self._client = self.loader(llm=self, **kwargs)

    def release_model(self):
        del self._client

    @property
    def ctx_size(self):
        ctx_size_map = {
            "mistral-large-latest": 128_000,
            "open-mistral-nemo": 128_000,
            "codestral-latest": 32_000,
            "open-mistral-7b": 32_000,
            "open-mixtral-8x7b": 32_000,
            "open-mixtral-8x22b": 64_000,
            "codestral-mamba": 256_000,
        }

        if self.name in ctx_size_map:
            return ctx_size_map[self.name]
        else:
            #: we don't know the model, so we'll default
            #: to a large context window of 128k tokens
            return 128_000
