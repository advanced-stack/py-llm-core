# -*- coding: utf-8 -*-
import json
import codecs
import openai

from .core import BaseParser


class OpenAIAssistant(BaseParser):
    def __init__(self, target_cls, model="gpt-3.5-turbo", *args, **kwargs):
        super().__init__(target_cls, *args, **kwargs)
        self.model = model

    def max_supported_tokens(self):
        if self.model == "gpt-3.5-turbo":
            return 4_000
        elif self.model == "gpt-3.5-turbo-16k":
            return 16_000
        elif self.model == "gpt-4":
            return 8_000
        elif self.model == "gpt-4-32k":
            return 32_000
        else:
            raise KeyError("Unsupported model")

    def process(self, **kwargs):
        system_prompt = getattr(self.target_cls, "system_prompt", "")
        prompt = getattr(self.target_cls, "prompt", "")

        rendered_system_prompt = self.target_cls.system_prompt.format(**kwargs)
        rendered_prompt = self.target_cls.prompt.format(**kwargs)

        json_schema = json.dumps(self.schema)
        complete_prompt = "\n".join((prompt, system_prompt, json_schema))
        token_count = len(codecs.encode(complete_prompt, self.model))
        model_window_size = self.max_supported_tokens()

        if token_count > model_window_size:
            raise ValueError(
                f"Too many tokens required {token_count} > {model_window_size}"
            )

        completion = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": rendered_system_prompt,
                },
                {
                    "role": "user",
                    "content": rendered_prompt,
                },
            ],
            functions=[self.schema],
            function_call={"name": "PublishAnswer"},
            temperature=0,
        )

        response = completion["choices"][0]["message"]["function_call"][
            "arguments"
        ]

        instance = self.deserialize(response)

        return instance
