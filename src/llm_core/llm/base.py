# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import List
import codecs
import json


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
            schema_prompt = json.dumps(schema_prompt)

        complete_prompt = [
            self.system_prompt,
            prompt,
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
    content: str
    function_call: dict = None
    tool_calls: dict = None


@dataclass
class ChatCompletionChoice:
    index: int
    message: Message
    finish_reason: str

    @classmethod
    def from_iterable(cls, iterable):
        for item in iterable:
            message_attrs = item["message"]

            # To ensure compatibility with other models, we stuff
            # function calling in the message content.

            function_call = message_attrs.get("function_call")
            if function_call:
                message_attrs["content"] = message_attrs["function_call"][
                    "arguments"
                ]

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
    choices: List[ChatCompletionChoice]
    usage: Usage
    system_fingerprint: str = None
    prompt_filter_results: dict = None

    @classmethod
    def parse(cls, attrs):
        attributes = {}
        attributes.update(attrs)

        attributes["choices"] = list(
            ChatCompletionChoice.from_iterable(attributes["choices"])
        )
        attributes["usage"] = Usage(**attributes["usage"])

        return cls(**attributes)
