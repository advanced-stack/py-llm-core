# -*- coding: utf-8 -*-
from openai import OpenAI
from typing import Callable
from dataclasses import dataclass

from .base import LLMBase


def create_completion(
    llm,
    model,
    messages,
    temperature,
    tools=None,
    tool_choice=None,
    schema=None,
):
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        tools=tools,
        tool_choice=tool_choice,
    )

    return completion.dict()


@dataclass
class OpenAIChatModel(LLMBase):
    create_completion: Callable = create_completion

    @property
    def ctx_size(self):
        ctx_size_map = {
            "gpt-3.5-turbo": 4_000,
            "gpt-3.5-turbo-0613": 4_000,
            "gpt-3.5-turbo-16k": 16_000,
            "gpt-3.5-turbo-16k-0613": 16_000,
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
            raise KeyError("Unsupported model")
