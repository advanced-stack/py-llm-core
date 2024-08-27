# -*- coding: utf-8 -*-
import json
import textwrap
import logging
import dirtyjson

from dataclasses import dataclass
from typing import Callable

from .llm import (
    OpenAIChatModel,
    MistralAIModel,
    OpenWeightsModel,
    AnthropicModel,
)
from .schema import to_json_schema, from_dict


@dataclass
class BaseParser:
    target_cls: Callable
    model: str
    model_cls: Callable
    loader: Callable = None
    loader_kwargs: dict = None
    system_prompt: str = (
        "Parse and process information from unstructured content."
    )

    def __post_init__(self):
        self.target_json_schema = to_json_schema(self.target_cls)

        llm_kwargs = {}

        if self.loader:
            llm_kwargs["loader"] = self.loader

        if self.loader_kwargs:
            llm_kwargs["loader_kwargs"] = self.loader_kwargs

        self.llm = self.model_cls(
            name=self.model, system_prompt=self.system_prompt, **llm_kwargs
        )

    def __enter__(self):
        self.llm.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.llm.release_model()

    def deserialize(self, json_str):
        attributes = dirtyjson.loads(json_str)
        return from_dict(self.target_cls, attributes)

    def parse(self, text):
        prompt = textwrap.dedent(
            """
            Carefully extract and parse all information available from the
            content while observing the expected schema hereunder:

            {formatted_schema}

            Write the answer using JSON.
            """.format(
                formatted_schema=json.dumps(self.target_json_schema, indent=2)
            )
        )
        history = [{"role": "user", "content": text}]

        completion = self.llm.ask(
            prompt, history, schema=self.target_json_schema
        )

        try:
            instance = self.deserialize(completion.choices[0].message.content)
        except Exception as e:
            error_message = "\n".join(
                (
                    "Unable to parse the following string:",
                    repr(completion.choices[0].message.content),
                )
            )
            logging.exception(error_message)
            raise e

        return instance


@dataclass
class OpenAIParser(BaseParser):
    target_cls: Callable
    model: str = "gpt-4o-mini"
    model_cls: Callable = OpenAIChatModel


@dataclass
class MistralAIParser(BaseParser):
    target_cls: Callable
    model: str = "open-mistral-nemo"
    model_cls: Callable = MistralAIModel


@dataclass
class OpenWeightsParser(BaseParser):
    target_cls: Callable
    model: str = "mistral-7b-v0.3-q4"
    model_cls: Callable = OpenWeightsModel


@dataclass
class AnthropicParser(BaseParser):
    target_cls: Callable
    model: str = "claude-3-5-sonnet-20240620"
    model_cls: Callable = AnthropicModel
