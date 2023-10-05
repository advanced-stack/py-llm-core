# -*- coding: utf-8 -*-
import json
import llama_cpp
import platform

from .core import BaseParser
from .llm import OpenAIChatModel


class OpenAIParser(BaseParser):
    def __init__(self, target_cls, model="gpt-3.5-turbo", *args, **kwargs):
        super().__init__(target_cls, *args, **kwargs)
        self.model = OpenAIChatModel(
            model=model,
            system_prompt=(
                "Act as a powerful AI able to extract, parse and process "
                "information from unstructured content."
            ),
        )

    def parse(self, text):
        prompt = "Extract and parse information from provided content"

        history = [
            {
                "role": "user",
                "content": text,
            },
        ]
        completion = self.ask(
            prompt,
            history,
            functions=[self.schema],
            function_call={"name": "PublishAnswer"},
        )

        response = completion["choices"][0]["message"]["function_call"][
            "arguments"
        ]

        instance = self.deserialize(response)

        return instance


class LLaMACPPParser(BaseParser):
    """
    Parser using the https://github.com/ggerganov/llama.cpp library

    Not limited to LLaMA models. For the complete list of supported models,

    see https://github.com/ggerganov/llama.cpp#readme
    """

    def __init__(
        self, target_cls, model_path, llama_cpp_kwargs=None, *args, **kwargs
    ):
        super().__init__(target_cls, *args, **kwargs)

        if llama_cpp_kwargs is None:
            llama_cpp_kwargs = {
                "n_ctx": 4000,
                "verbose": False,
            }

            if platform.system() == "Darwin" and platform.machine() == "arm64":
                llama_cpp_kwargs["n_gpu_layers"] = 33

        self.model = llama_cpp.Llama(model_path, **llama_cpp_kwargs)

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.model

    def parse(self, text):
        completion = self.model(
            text,
            temperature=0.1,
            mirostat_mode=2,
            max_tokens=4000,  #: TODO: Compute prompt size and adapt max token
            grammar=self.grammar,
        )

        response = completion["choices"][0]["text"]
        instance = self.deserialize(response)
        return instance
