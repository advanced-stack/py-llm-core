# -*- coding: utf-8 -*-
from dataclasses import dataclass, fields
import codecs
import json


def remove_unsupported_attributes(data_cls, attributes):
    field_names = set(field.name for field in fields(data_cls))
    return {k: v for k, v in attributes.items() if k in field_names}


@dataclass
class LLMBase:
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

        required_ctx_size = len(codecs.encode(complete_prompt, self.name))
        if required_ctx_size > self.ctx_size:
            raise OverflowError(
                f"Prompt too large {required_ctx_size} for this model {self.ctx_size}"
            )

        return self.ctx_size - required_ctx_size


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

            message = Message(**item["message"])
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
