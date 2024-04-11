# -*- coding: utf-8 -*-
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

from dataclasses import dataclass

from .base import (
    LLMBase,
    ChatCompletion,
)

from ..settings import (
    MISTRAL_API_KEY,
    USE_AZURE_AI_MISTRAL_LARGE,
    AZURE_AI_MISTRAL_LARGE_ENDPOINT,
    AZURE_AI_MISTRAL_LARGE_KEY,
)


def create_chat_completion(
    model, messages, temperature, max_tokens=1000, **kwargs
):
    extra_args = {}
    if USE_AZURE_AI_MISTRAL_LARGE:
        endpoint = AZURE_AI_MISTRAL_LARGE_ENDPOINT
        api_key = AZURE_AI_MISTRAL_LARGE_KEY
        model = "azureai"
        extra_args["endpoint"] = endpoint
    else:
        api_key = MISTRAL_API_KEY
        model = "mistral-large-latest"

    client = MistralClient(api_key=api_key, **extra_args)

    completion = client.chat(model=model, messages=messages, **kwargs)

    return completion


@dataclass
class MistralAILarge(LLMBase):
    name: str = "mistral-large-latest"
    system_prompt: str = "You are a helpful assistant"

    def __enter__(self):
        # No special initialization required as we are using API
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # No special cleanup required as we are using API
        pass

    @property
    def ctx_size(self):
        return 32_000

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

        messages = [
            ChatMessage(
                **{
                    "role": "system",
                    "content": self.system_prompt,
                }
            ),
        ]
        if history:
            messages += [ChatMessage(**msg) for msg in history]

        messages.append(
            ChatMessage(
                **{
                    "role": "user",
                    "content": prompt,
                }
            ),
        )

        kwargs = {}
        if schema:
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "PublishAnswer",
                        "description": "Publish the answer",
                        "parameters": schema,
                    },
                }
            ]
            tool_choice = "any"
            kwargs = {
                "tools": tools,
                "tool_choice": tool_choice,
            }

        completion = create_chat_completion(
            model=self.name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        return ChatCompletion.parse(completion.dict())
