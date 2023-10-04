# -*- coding: utf-8 -*-
import json
import codecs
import openai
import llama_cpp

from .core import BaseParser


class OpenAIParser(BaseParser):
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

    def parse(self, text):
        prompt = "Extract and parse information from provided content"
        system_prompt = (
            "Act as a powerful AI able to extract, parse and process "
            "information from unstructured content"
        )

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
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": text,
                },
                {
                    "role": "user",
                    "content": prompt,
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


class LLamaParser(BaseParser):
    def __init__(
        self, target_cls, model_path, llama_cpp_kwargs=None, *args, **kwargs
    ):
        super().__init__(target_cls, *args, **kwargs)

        if llama_cpp_kwargs is None:
            llama_cpp_kwargs = {
                "n_ctx": 8000,
                "n_gpu_layers": 33,
                "verbose": False,
            }
        self.model = llama_cpp.Llama(model_path, **llama_cpp_kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.model

    def parse(self, text):
        completion = self.model(
            text,
            temperature=0.1,
            mirostat_mode=2,
            max_tokens=8000,  #: TODO: Compute prompt size and adapt max token
            grammar=self.grammar,
        )

        response = completion["choices"][0]["text"]
        instance = self.deserialize(response)
        return instance
