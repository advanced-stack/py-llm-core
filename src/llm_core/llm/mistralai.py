# -*- coding: utf-8 -*-
import httpx
from openai import AzureOpenAI, OpenAI

import os

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

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

def create_chat_completion_mistral(model, messages, temperature, max_tokens=1000, **kwargs):
   
    api_key = os.environ["MISTRAL_API_KEY"]
    print(messages)
    client = MistralClient(api_key=api_key)
    print(model)
    # Appel à l'API Mistral pour générer une complétion
    completion = client.chat(
        model=model,
        messages=messages,
        temperature=temperature, 
        **kwargs,
        )

    return completion


@dataclass
class MistralAiChatModel(LLMBase):
    name: str = "mistral-tiny"
   
    system_prompt: str = "You are a helpful assistant"

    def __enter__(self):
        # No special initialization required as we are using API
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # No special cleanup required as we are using API
        pass

    @property
    def ctx_size(self):
        if self.name == "mistral-tiny":
            return 32_000

    def ask(
    self,
    prompt,
    history=None,
    schema=None,
    temperature=0,
    ):
       
        messages=[]

        if history:
            for msg in history:
                messages.append(ChatMessage(role="system", content=msg))

        messages.append(ChatMessage(role="user", content=prompt))
        kwargs = {}
        if schema:

                    tools = [
                                {
                                    "type": "function",
                                    "function": {
                                        "name": "PublishAnswer",
                                        "description": "Publish the answer",
                                        "parameters": schema

                                    },
                                }
                            ]
                    tool_choice ="auto"
                    kwargs = {
                            "tools": tools,
                            "tool_choice": tool_choice,
                        }

        completion = create_chat_completion_mistral(
        model=self.name,
        messages=messages,
        temperature=temperature,
        **kwargs,
        
        )
        return ChatCompletion.parse(completion.dict())