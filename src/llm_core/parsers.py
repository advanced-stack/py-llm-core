# -*- coding: utf-8 -*-
import json
import textwrap
import logging
import dirtyjson

from .llm import (
    OpenAIChatModel,
    MistralAIModel,
    OpenWeightsModel,
    load_model,
)
from .schema import to_json_schema, from_dict


class BaseParser:
    def __init__(self, target_cls, *args, **kwargs):
        self.target_cls = target_cls
        self.target_json_schema = to_json_schema(self.target_cls)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def deserialize(self, json_str):
        attributes = dirtyjson.loads(json_str)
        return from_dict(self.target_cls, attributes)

    def parse(self, text):
        prompt = textwrap.dedent(
            """
            Carefully extract and parse all information available from the
            content while observing the expected schema hereunder:

            {formatted_schema}

            When fields are written in plural, be sure to include all instances
            you found.
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


class OpenAIParser(BaseParser):
    def __init__(self, target_cls, model="gpt-4o-mini", *args, **kwargs):
        super().__init__(target_cls, *args, **kwargs)
        self.llm = OpenAIChatModel(
            name=model,
            system_prompt=(
                "Parse and process information from unstructured content."
            ),
        )


class MistralAIParser(BaseParser):
    def __init__(self, target_cls, model="open-mistral-nemo", *args, **kwargs):
        super().__init__(target_cls, *args, **kwargs)
        self.llm = MistralAIModel(
            name=model,
            system_prompt=(
                "Parse and process information from unstructured content."
            ),
        )


class OpenWeightsParser(BaseParser):
    def __init__(
        self,
        target_cls,
        model="mistral-7b-v0.3-q4",
        model_loader=load_model,
        model_loader_kwargs=None,
        *args,
        **kwargs
    ):
        super().__init__(target_cls, *args, **kwargs)
        self.llm = OpenWeightsModel(
            name=model,
            model_loader=model_loader,
            model_loader_kwargs=model_loader_kwargs,
        )

    def __enter__(self):
        self.llm.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.llm.release_model()
