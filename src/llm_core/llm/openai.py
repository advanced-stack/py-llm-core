# -*- coding: utf-8 -*-
import httpx
from openai import AzureOpenAI, OpenAI

from dataclasses import dataclass

from .base import (
    LLMBase,
    ChatCompletion,
)

from ..settings import (
    USE_AZURE_OPENAI,
    AZURE_OPENAI_ENDPOINT,
    DEFAULT_TIMEOUT,
)


def create_chat_completion(
    model, messages, temperature, max_tokens=1000, **kwargs
):
    default_timeout = httpx.Timeout(DEFAULT_TIMEOUT, write=10.0, connect=2.0)

    if USE_AZURE_OPENAI:
        # gets the API Key from environment variable AZURE_OPENAI_API_KEY
        client = AzureOpenAI(
            api_version="2023-09-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            timeout=default_timeout,
        )

        # Here we are using the following convention:
        # Every model name is mapped to an Azure deployment where we remove the
        # dot sign.
        model_name = model.replace(".", "")
    else:
        client = OpenAI(timeout=default_timeout)
        model_name = model

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )

    return completion


@dataclass
class OpenAIChatModel(LLMBase):
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
        if self.name in ("gpt-3.5-turbo", "gpt-3.5-turbo-0613"):
            return 4_000
        elif self.name in ("gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613"):
            return 16_000
        elif self.name in ("gpt-4", "gpt-4-0613"):
            return 8_000
        elif self.name == "gpt-4-32k":
            return 32_000
        elif self.name == "gpt-4-1106-preview":
            return 128_000
        else:
            raise KeyError("Unsupported model")

    def ask(
        self,
        prompt,
        history=None,
        schema=None,
        temperature=0,
    ):
        max_tokens = self.sanitize_prompt(
            prompt=prompt,
            history=history,
            schema=schema,
        )

        #: Reduce by 10 percent the maximum tokens to be generated to take into
        #: account inaccuracies of sanitize_prompt (especially the schema token
        #: consumption)

        max_tokens = int(0.9 * max_tokens)

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
        ]
        if history:
            messages += history

        messages.append(
            {
                "role": "user",
                "content": prompt,
            },
        )

        kwargs = {}
        if schema:
            functions = {
                "name": "PublishAnswer",
                "description": "Publish the answer",
                "parameters": schema,
            }
            function_call = {"name": "PublishAnswer"}

            kwargs = {
                "functions": [functions],
                "function_call": function_call,
            }

        completion = create_chat_completion(
            model=self.name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return ChatCompletion.parse(completion.dict())
