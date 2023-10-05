# -*- coding: utf-8 -*-
import platform
import llama_cpp
import codecs
import json

from dataclasses import dataclass


@dataclass
class LLaMACPPModel:
    """
    Model wrapper using the https://github.com/ggerganov/llama.cpp library

    Not limited to LLaMA models. For the complete list of supported models,

    see https://github.com/ggerganov/llama.cpp#readme
    """

    name: str = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
    system_prompt: str = "You are a helpful assistant"
    ctx: int = 4096
    verbose: bool = False

    def __enter__(self, **llama_cpp_kwargs):
        if not llama_cpp_kwargs:
            llama_cpp_kwargs = {
                "n_ctx": self.ctx,
                "verbose": self.verbose,
            }

            if platform.system() == "Darwin" and platform.machine() == "arm64":
                # Offload everything onto the GPU on MacOS
                llama_cpp_kwargs["n_gpu_layers"] = 1000

        self.model = llama_cpp.Llama(self.name, **llama_cpp_kwargs)

        return self

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

    def sanitize_prompt(
        self, prompt, history=None, functions=None, function_call=None
    ):
        functions_prompt = ""
        function_call_prompt = ""

        if functions:
            functions_prompt = json.dumps(functions)

        if function_call:
            function_call_prompt = json.dumps(function_call)

        complete_prompt = [
            self.system_prompt,
            prompt,
            functions_prompt,
            function_call_prompt,
        ]

        complete_prompt = "\n".join(complete_prompt)

        ctx_size = len(codecs.encode(complete_prompt, self.name))
        if ctx_size > self.ctx_size:
            raise OverflowError(
                f"Prompt too large {ctx_size} for this model {self.ctx_size}"
            )

    def ask(
        self,
        prompt,
        history=None,
        functions=None,
        function_call=None,
        temperature=0,
    ):
        self.sanitize_prompt(
            prompt=prompt,
            history=history,
            functions=functions,
            function_call=function_call,
        )
        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
        ]
        if history:
            messages.append(history)

        messages.append(
            {
                "role": "user",
                "content": prompt,
            },
        )

        kwargs = {}
        if functions:
            kwargs = {
                "functions": functions,
                "function_call": function_call,
            }

        completion = openai.ChatCompletion.create(
            model=self.name,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return completion
