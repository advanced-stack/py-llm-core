# -*- coding: utf-8 -*-
from anthropic import Anthropic

from dataclasses import dataclass
from typing import Callable

from .base import LLMBase
from ..settings import ANTHROPIC_API_KEY


def load_anthropic_client(llm, **kwargs):
    client = Anthropic(api_key=ANTHROPIC_API_KEY, **kwargs)
    return client


def map_tools(tools):
    mapped_tools = []
    for tool in tools:
        if tool.get("type") == "function" and "function" in tool:
            function_info = tool["function"]
            mapped_tool = {
                "name": function_info.get("name"),
                "description": function_info.get("description"),
                "input_schema": function_info.get("parameters"),
            }
            mapped_tools.append(mapped_tool)
    return mapped_tools


def create_anthropic_completion(
    llm,
    model,
    messages,
    temperature,
    tools=None,
    tool_choice=None,
    schema=None,
):
    kwargs = {}
    if tools:
        kwargs["tools"] = map_tools(tools)

    if tool_choice:
        kwargs["tool_choice"] = {"type": "any"}

    system_message = "\n".join(
        (
            item["content"]
            for item in filter(lambda m: m["role"] == "system", messages)
        )
    )
    messages = list(filter(lambda m: m["role"] != "system", messages))

    # Merge messages with subsequent identical roles
    merged_messages = []
    for message in messages:
        if merged_messages and merged_messages[-1]["role"] == message["role"]:
            merged_messages[-1]["content"] += "\n" + message["content"]
        else:
            merged_messages.append(message)

    completion = llm._client.messages.create(
        model=model,
        messages=merged_messages,
        max_tokens=2_000,
        system=system_message,
        **kwargs,
    )
    return completion.to_dict()


@dataclass
class AnthropicModel(LLMBase):
    create_completion: Callable = create_anthropic_completion
    loader: Callable = load_anthropic_client
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
            "claude-3-5-sonnet-20240620": 200_000,
            "claude-3-opus-20240229": 200_000,
        }

        if self.name in ctx_size_map:
            return ctx_size_map[self.name]
        else:
            #: we don't know the model, so we'll default
            #: to a large context window of 200k tokens
            return 200_000
