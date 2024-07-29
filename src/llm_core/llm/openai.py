# -*- coding: utf-8 -*-
import httpx
import traceback
import dirtyjson

from openai import AzureOpenAI, OpenAI
from dataclasses import dataclass

from .base import (
    LLMBase,
    ChatCompletion,
)

from ..schema import (
    from_dict,
    make_selection_tool,
)

from ..settings import (
    USE_AZURE_OPENAI,
    AZURE_OPENAI_ENDPOINT,
    DEFAULT_TIMEOUT,
)


def as_tool(json_schema):
    return {
        "type": "function",
        "function": {
            "name": json_schema["title"],
            "description": json_schema["description"],
            "parameters": json_schema,
        },
    }


def _generate_completion(
    model,
    messages,
    temperature,
    tools=None,
    tool_choice=None,
):
    default_timeout = httpx.Timeout(DEFAULT_TIMEOUT, write=10.0, connect=2.0)

    if USE_AZURE_OPENAI:
        # gets the API Key from environment variable AZURE_OPENAI_API_KEY
        client = AzureOpenAI(
            api_version="2024-02-01",
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
        tools=tools,
        tool_choice=tool_choice,
    )

    return ChatCompletion.parse(completion.dict())


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
        elif self.name in (
            "gpt-4-1106-preview",
            "gpt-4o-2024-05-13",
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-mini",
        ):
            return 128_000
        else:
            raise KeyError("Unsupported model")

    def ask(
        self,
        prompt,
        history=(),
        schema=None,
        temperature=0,
        tools=None,
    ):
        self.sanitize_prompt(
            prompt=prompt,
            history=history,
            schema=schema,
        )

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

        if tools:
            tool_selector = make_selection_tool(tools)
            tools = [as_tool(tool_selector.schema)]

            system_prompt_override = (
                "Act deterministic and take logical steps."
                "Use only available tools."
            )
            messages = [
                {
                    "role": "system",
                    "content": system_prompt_override,
                },
                {
                    "role": "system",
                    "content": system_prompt_override,
                },
                *history,
                {
                    "role": "assistant",
                    "content": tool_selector.helpers,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
                {
                    "role": "assistant",
                    "content": tool_selector.prompt,
                },
            ]

            completion = _generate_completion(
                model=self.name,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice={
                    "type": "function",
                    "function": {"name": tool_selector.__name__},
                },
            )

            attributes = dirtyjson.loads(completion.choices[0].message.content)

            instance = from_dict(tool_selector, attributes)

            try:
                result = instance.execute()
            except Exception as e:
                traceback.print_exc()
                result = repr(e)

            messages.append(
                {
                    "role": "assistant",
                    "content": instance.detailed_plan.format_results(result),
                },
            )

            messages.append(
                {
                    "role": "assistant",
                    "content": "Answer very concisely to the user's query",
                },
            )

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
            tool_choice = {
                "type": "function",
                "function": {"name": "PublishAnswer"},
            }

            return _generate_completion(
                model=self.name,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
            )

        return _generate_completion(
            model=self.name,
            messages=messages,
            temperature=temperature,
        )
