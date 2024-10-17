# -*- coding: utf-8 -*-
import uuid
import requests
from dataclasses import dataclass
from typing import Callable

from .base import LLMBase
from ..settings import GOOGLE_API_KEY


def map_response_to_chat_completion_attributes(model, response):
    # Extract candidates and usage metadata
    candidates = response.get("candidates", [])
    usage_metadata = response.get("usageMetadata", {})

    # Prepare choices
    choices = []
    for candidate in candidates:
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        text = parts[0].get("text", "") if parts else ""
        role = content.get("role", "model")

        message = {"role": role, "content": text}
        choice = {
            "index": candidate.get("index", 0),
            "message": message,
            "finish_reason": candidate.get("finishReason", ""),
        }
        choices.append(choice)

    # Prepare usage
    usage = {
        "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
        "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
        "total_tokens": usage_metadata.get("totalTokenCount", 0),
    }

    # Create attributes dictionary for ChatCompletion
    chat_completion_attributes = {
        "id": uuid.uuid4().hex,  # Generate a unique ID
        "model": model,  # Assuming a model name, adjust as needed
        "usage": usage,
        "choices": choices,
    }

    return chat_completion_attributes


def strip_titles(schema):
    schema.pop("title", None)

    properties = schema.get("properties", None)
    if properties is not None:
        for name, value in properties.items():
            strip_titles(value)

    items = schema.get("items", None)
    if items is not None:
        strip_titles(items)


def load_google_ai_client(llm, **kwargs):
    client_kwargs = {}
    client_kwargs.update(kwargs)

    api_key = client_kwargs.pop("api_key", GOOGLE_API_KEY)
    base_url = client_kwargs.pop(
        "base_url", "https://generativelanguage.googleapis.com/v1beta/models/"
    )
    endpoint = client_kwargs.pop("endpoint", ":generateContent")
    headers = client_kwargs.pop(
        "headers", {"Content-Type": "application/json"}
    )

    return requests.Request(
        "POST",
        f"{base_url}{llm.name}{endpoint}",
        headers=headers,
        params={"key": api_key},
    )


def extract_system_messages(messages):
    return map(
        lambda m: m["content"],
        filter(lambda m: m["role"] == "system", messages),
    )


def extract_conversation(messages):
    return filter(lambda m: m["role"] in ["user", "assistant"], messages)


def format_message(message):
    role_mapping = {
        "assistant": "model",
        "user": "user",
    }
    return {
        "role": role_mapping[message["role"]],
        "parts": [{"text": message["content"]}],
    }


def create_google_ai_completion(
    llm,
    model,
    messages,
    temperature,
    tools=None,
    tool_choice=None,
    schema=None,
):
    system_messages = extract_system_messages(messages)
    conversation = extract_conversation(messages)
    contents = map(format_message, conversation)

    payload = {
        "system_instruction": {"parts": {"text": "\n".join(system_messages)}},
        "contents": list(contents),
    }

    if schema:
        strip_titles(schema)

        payload.update(
            {
                "generationConfig": {
                    "response_mime_type": "application/json",
                    "response_schema": schema,
                }
            }
        )

    llm._client.json = payload

    request = llm._client.prepare()

    session = requests.Session()
    response = session.send(request)

    return map_response_to_chat_completion_attributes(
        llm.name, response.json()
    )


@dataclass
class GoogleAIModel(LLMBase):
    create_completion: Callable = create_google_ai_completion
    loader: Callable = load_google_ai_client
    loader_kwargs: dict = None

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_model()

    def load_model(self):
        kwargs = self.loader_kwargs or {}
        self._client = self.loader(llm=self, **kwargs)

    def release_model(self):
        del self._client

    @property
    def ctx_size(self):
        ctx_size_map = {
            "gemini-1.5-flash": 1_048_576,
        }

        if self.name in ctx_size_map:
            return ctx_size_map[self.name]
        else:
            #: we don't know the model, so we'll default
            #: to a large context window of 128k tokens
            return 128_000
