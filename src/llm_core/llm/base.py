# -*- coding: utf-8 -*-
import json
import codecs
import traceback
import dirtyjson

from typing import Callable
from dataclasses import dataclass, fields

from ..schema import (
    as_tool,
    from_dict,
    make_selection_tool,
)


@dataclass
class LLMBase:
    name: str = "model-name"
    system_prompt: str = "You are a helpful assistant"
    create_completion: Callable = None
    loader: Callable = None
    loader_kwargs: dict = None

    def __post_init__(self):
        self._ctx_size = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @property
    def ctx_size(self):
        return self._ctx_size

    @ctx_size.setter
    def ctx_size(self, value):
        self._ctx_size = value

    def _generate_completion(
        self,
        model,
        messages,
        temperature,
        tools=None,
        tool_choice=None,
        schema=None,
    ):
        completion = self.create_completion(
            llm=self,
            model=model,
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice=tool_choice,
            schema=schema,
        )
        return ChatCompletion.parse(completion)

    def ask(self, prompt, history=(), schema=None, temperature=0, tools=None):
        self.sanitize_prompt(prompt=prompt, history=history, schema=schema)

        messages = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages += history

        messages.append({"role": "user", "content": prompt})

        if tools:
            tool_selector = make_selection_tool(tools)
            tools = [as_tool(tool_selector.schema)]

            system_prompt_override = (
                "Act deterministic and take logical steps."
                "Use only available tools."
            )
            messages = [
                {"role": "system", "content": system_prompt_override},
                {"role": "system", "content": system_prompt_override},
                *history,
                {"role": "assistant", "content": tool_selector.helpers},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": tool_selector.prompt},
            ]

            completion = self._generate_completion(
                model=self.name,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice={
                    "type": "function",
                    "function": {"name": tool_selector.__name__},
                },
                schema=tool_selector.schema,
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
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "Answer concisely to the user's query (in the same language)",
                }
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

            return self._generate_completion(
                model=self.name,
                messages=messages,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
                schema=schema,
            )

        return self._generate_completion(
            model=self.name, messages=messages, temperature=temperature
        )

    def sanitize_prompt(
        self,
        prompt,
        history=None,
        schema=None,
    ):
        schema_prompt = ""

        if schema:
            schema_prompt = json.dumps(schema)

        complete_prompt = [
            self.system_prompt,
            prompt,
            str(history) if history else "",
            schema_prompt,
        ]

        complete_prompt = "\n".join(complete_prompt)

        prompt_len = len(codecs.encode(complete_prompt, "tiktoken"))

        if prompt_len > self.ctx_size:
            err = (
                f"Prompt too large {prompt_len} for this model {self.ctx_size}"
            )
            raise OverflowError(err)

        return self.ctx_size - prompt_len


def remove_unsupported_attributes(data_cls, attributes):
    field_names = set(field.name for field in fields(data_cls))
    return {k: v for k, v in attributes.items() if k in field_names}


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Message:
    role: str
    content: str = None
    function_call: dict = None
    tool_calls: list = None
    name: str = None


@dataclass
class ChatCompletionChoice:
    index: int
    message: Message
    finish_reason: str

    @classmethod
    def from_iterable(cls, iterable):
        for item in iterable:
            message_attrs = item["message"]

            function_call = message_attrs.get("function_call")
            tool_calls = message_attrs.get("tool_calls")
            if function_call:
                message_attrs["content"] = message_attrs["function_call"][
                    "arguments"
                ]
            elif tool_calls:
                message_attrs["content"] = message_attrs["tool_calls"][0][
                    "function"
                ]["arguments"]

            attrs = remove_unsupported_attributes(Message, item["message"])
            message = Message(**attrs)
            index = item["index"]
            finish_reason = item["finish_reason"]

            yield cls(
                index=index, message=message, finish_reason=finish_reason
            )


@dataclass
class ChatCompletion:
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage
    system_fingerprint: str = None
    prompt_filter_results: dict = None

    @classmethod
    def parse(cls, attrs):
        attributes = remove_unsupported_attributes(cls, attrs)

        attributes["choices"] = list(
            ChatCompletionChoice.from_iterable(attributes["choices"])
        )
        attributes["usage"] = Usage(**attributes["usage"])

        return cls(**attributes)
