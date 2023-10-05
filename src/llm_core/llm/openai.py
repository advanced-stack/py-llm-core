# -*- coding: utf-8 -*-
import json
import codecs
import openai

from dataclasses import dataclass


@dataclass
class OpenAIChatModel:
    name: str = "gpt-3.5-turbo"
    system_prompt: str = "You are a helpful assistant"

    def __enter__(self):
        # No special initialization required as we are using API
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # No special cleanup required as we are using API
        pass

    @property
    def ctx_size(self):
        if self.name == "gpt-3.5-turbo":
            return 4_000
        elif self.name == "gpt-3.5-turbo-16k":
            return 16_000
        elif self.name == "gpt-4":
            return 8_000
        elif self.name == "gpt-4-32k":
            return 32_000
        else:
            raise KeyError("Unsupported model")

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
