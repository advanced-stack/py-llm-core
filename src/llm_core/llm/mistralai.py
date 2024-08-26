# -*- coding: utf-8 -*-
from mistralai import Mistral
from dataclasses import dataclass
from typing import Callable

from .base import LLMBase
from ..settings import MISTRAL_API_KEY


def create_completion(
    llm,
    model,
    messages,
    temperature,
    tools=None,
    tool_choice=None,
    schema=None,
):
    api_key = MISTRAL_API_KEY
    client = Mistral(api_key=api_key)

    # As per Mistral Documentation to force usage of tool.
    if tool_choice:
        tool_choice = "any"

    completion = client.chat.complete(
        model=model, messages=messages, tools=tools, tool_choice=tool_choice
    )
    return completion.dict()


@dataclass
class MistralAIModel(LLMBase):
    create_completion: Callable = create_completion

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
            raise KeyError("Unsupported model")
