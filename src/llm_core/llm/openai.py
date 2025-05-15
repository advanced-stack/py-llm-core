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
    client_kwargs = {}
    client_kwargs.update(kwargs)

    api_key = client_kwargs.pop("api_key", AZURE_OPENAI_API_KEY)
    api_version = client_kwargs.pop("api_version", AZURE_OPENAI_API_VERSION)
    azure_endpoint = client_kwargs.pop("azure_endpoint", AZURE_OPENAI_ENDPOINT)

    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        **client_kwargs,
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

    if tools and "gpt-4o" in model:
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
            "gpt-4.1": 1_000_000,
            "gpt-4.1-mini": 1_000_000,
            "gpt-4.1-nano": 1_000_000,
        }
        return ctx_size_map.get(self.name, 128_000)

    @property
    def output_ctx_size(self):
        output_estimates = {
            "gpt-3.5-turbo": 4_096,
            "gpt-3.5-turbo-0613": 4_096,
            "gpt-3.5-turbo-16k": 16_384,
            "gpt-3.5-turbo-16k-0613": 16_384,
            "gpt-35-turbo": 4_096,
            "gpt-35-turbo-16k": 16_384,
            "gpt-4": 8_192,
            "gpt-4-0613": 8_192,
            "gpt-4-32k": 32_768,
            "gpt-4-1106-preview": 8_192,
            "gpt-4o-2024-05-13": 16_384,
            "gpt-4o-2024-08-06": 16_384,
            "gpt-4o": 16_384,
            "gpt-4o-mini-2024-07-18": 16_384,
            "gpt-4o-mini": 16_384,
            "gpt-4.1": 32_768,
            "gpt-4.1-mini": 32_768,
            "gpt-4.1-nano": 32_768,
        }
        return output_estimates.get(self.name, 4_096)


@dataclass
class AzureOpenAIChatModel(OpenAIChatModel):
    loader: Callable = load_azure_openai_client
