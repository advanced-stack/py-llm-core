# -*- coding: utf-8 -*-
from ..parsers import BaseParser, OpenAIParser, LLaMACPPParser


class BaseAssistant(BaseParser):
    def __init__(self, target_cls, *args, **kwargs):
        super().__init__(target_cls, *args, **kwargs)
        self.system_prompt = getattr(self.target_cls, "system_prompt", "")
        self.prompt = getattr(self.target_cls, "prompt", "")

    def process(self, **kwargs):
        system_prompt = self.system_prompt.format(**kwargs)
        prompt = self.prompt.format(**kwargs)

        self.model_wrapper.system_prompt = system_prompt

        completion = self.model_wrapper.ask(
            prompt, schema=self.target_json_schema, **self.completion_kwargs
        )
        instance = self.deserialize(completion.choices[0].message.content)
        return instance


class OpenAIAssistant(BaseAssistant, OpenAIParser):
    def __init__(self, target_cls, model="gpt-3.5-turbo", *args, **kwargs):
        super().__init__(target_cls, model=model, *args, **kwargs)


class LLaMACPPAssistant(BaseAssistant, LLaMACPPParser):
    def __init__(
        self,
        target_cls,
        model="mistral",
        llama_cpp_kwargs=None,
        *args,
        **kwargs
    ):
        super().__init__(
            target_cls,
            model=model,
            llama_cpp_kwargs=llama_cpp_kwargs,
            *args,
            **kwargs
        )
