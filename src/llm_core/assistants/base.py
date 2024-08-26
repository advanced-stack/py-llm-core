# -*- coding: utf-8 -*-
from textwrap import dedent
from ..llm import load_model
from ..parsers import (
    BaseParser,
    OpenAIParser,
    OpenWeightsParser,
    MistralAIParser,
)


class BaseAssistant(BaseParser):
    def __init__(self, target_cls, *args, **kwargs):
        super().__init__(target_cls, *args, **kwargs)
        self.system_prompt = getattr(self.target_cls, "system_prompt", "")
        self.prompt = getattr(self.target_cls, "prompt", "")

    def process(self, **kwargs):
        system_prompt = dedent(self.system_prompt.format(**kwargs))
        prompt = dedent(self.prompt.format(**kwargs))

        self.llm.system_prompt = system_prompt

        tools = getattr(self, "tools", None)
        completion = self.llm.ask(
            prompt, schema=self.target_json_schema, tools=tools
        )
        instance = self.deserialize(completion.choices[0].message.content)
        return instance


class OpenAIAssistant(BaseAssistant, OpenAIParser):
    def __init__(self, target_cls, model="gpt-4o-mini", *args, **kwargs):
        super().__init__(target_cls, model=model, *args, **kwargs)


class MistralAIAssistant(BaseAssistant, MistralAIParser):
    def __init__(self, target_cls, model="open-mistral-nemo", *args, **kwargs):
        super().__init__(target_cls, model=model, *args, **kwargs)


class OpenWeightsAssistant(BaseAssistant, OpenWeightsParser):
    def __init__(
        self,
        target_cls,
        model="mistral-7b-v0.3-q4",
        model_loader=load_model,
        model_loader_kwargs=None,
        *args,
        **kwargs
    ):
        super().__init__(
            target_cls,
            model="mistral-7b-v0.3-q4",
            model_loader=load_model,
            model_loader_kwargs=None,
            *args,
            **kwargs
        )
