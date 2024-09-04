# -*- coding: utf-8 -*-
from openai import OpenAI, AzureOpenAI
from typing import Callable
from dataclasses import dataclass

from .base import LLMBase
from ..settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_VERSION,
)


def load_openai_client(llm, **kwargs):
    client = OpenAI(**kwargs)
    return client


def load_azure_openai_client(llm, **kwargs):
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        **kwargs,
    )
    return client


def create_openai_completion(
    llm,
    model,
    messages,
    temperature,
    tools=None,
    tool_choice=None,
    schema=None,
):
    additional_kwargs = {}

    if tools:
        additional_kwargs.update({"parallel_tool_calls": False})

    completion = llm._client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        tools=tools,
        tool_choice=tool_choice,
        **additional_kwargs,
    )
    return completion.dict()


@dataclass
class OpenAIChatModel(LLMBase):
    create_completion: Callable = create_openai_completion
    loader: Callable = load_openai_client
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
            "gpt-3.5-turbo": 4_000,
            "gpt-3.5-turbo-0613": 4_000,
            "gpt-3.5-turbo-16k": 16_000,
            "gpt-3.5-turbo-16k-0613": 16_000,
            "gpt-35-turbo": 4_000,
            "gpt-35-turbo-16k": 16_000,
            "gpt-4": 8_000,
            "gpt-4-0613": 8_000,
            "gpt-4-32k": 32_000,
            "gpt-4-1106-preview": 128_000,
            "gpt-4o-2024-05-13": 128_000,
            "gpt-4o-2024-08-06": 128_000,
            "gpt-4o": 128_000,
            "gpt-4o-mini-2024-07-18": 128_000,
            "gpt-4o-mini": 128_000,
        }

        if self.name in ctx_size_map:
            return ctx_size_map[self.name]
        else:
            #: we don't know the model, so we'll default
            #: to a large context window of 128k tokens
            return 128_000


@dataclass
class AzureOpenAIChatModel(OpenAIChatModel):
    loader: Callable = load_azure_openai_client
