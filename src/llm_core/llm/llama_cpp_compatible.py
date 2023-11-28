# -*- coding: utf-8 -*-
import os
import platform
import llama_cpp

from dataclasses import dataclass

from .base import (
    LLMBase,
    ChatCompletion,
)
from ..schema import to_grammar
from ..settings import MODELS_CACHE_DIR


@dataclass
class LLaMACPPModel(LLMBase):
    """
    Model wrapper using the https://github.com/ggerganov/llama.cpp library

    Not limited to LLaMA models. For the complete list of supported models,

    see https://github.com/ggerganov/llama.cpp#readme
    """

    name: str = "mistral"
    system_prompt: str = "You are a helpful assistant"
    ctx_size: int = 8000
    verbose: bool = False
    llama_cpp_kwargs: dict = None

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_model()

    def release_model(self):
        del self.model

    def load_model(self):
        if self.llama_cpp_kwargs is None:
            self.llama_cpp_kwargs = {
                "n_ctx": self.ctx_size,
                "verbose": self.verbose,
            }

            if platform.system() == "Darwin" and platform.machine() == "arm64":
                # Offload everything onto the GPU on MacOS
                self.llama_cpp_kwargs["n_gpu_layers"] = 100
                self.llama_cpp_kwargs["n_threads"] = 1

        model_path = os.path.join(MODELS_CACHE_DIR, self.name)
        self.model = llama_cpp.Llama(model_path, **self.llama_cpp_kwargs)

    def ask(
        self,
        prompt,
        history=None,
        schema=None,
        temperature=0,
        **llama_cpp_kwargs,
    ):
        self.sanitize_prompt(prompt=prompt, history=history, schema=schema)

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
            grammar = to_grammar(schema)
            kwargs = {"grammar": grammar}

        # Allow to call `ask` and free up memory immediately
        model = getattr(self, "model", None)

        if model:
            completion = model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=-1,
                **kwargs,
            )
        else:
            with self as wrapper:
                completion = wrapper.model.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    **kwargs,
                )

        return ChatCompletion.parse(completion)


@dataclass
class LLaVACPPModel(LLaMACPPModel):
    def load_model(self):
        if self.llama_cpp_kwargs is None:
            self.llama_cpp_kwargs = {
                "n_ctx": self.ctx_size,
                "verbose": self.verbose,
            }

            if platform.system() == "Darwin" and platform.machine() == "arm64":
                # Offload everything onto the GPU on MacOS
                self.llama_cpp_kwargs["n_gpu_layers"] = 100
                self.llama_cpp_kwargs["n_threads"] = 1

        model_path = os.path.join(MODELS_CACHE_DIR, self.name)
        chat_format = "llava-1-5"
        chat_handler = llama_cpp.llama_chat_format.Llava15ChatHandler(
            clip_model_path=self.llama_cpp_kwargs["clip_model_path"]
        )

        self.model = llama_cpp.Llama(
            model_path,
            chat_format=chat_format,
            chat_handler=chat_handler,
            **self.llama_cpp_kwargs,
        )

    def ask(
        self,
        prompt,
        history=None,
        schema=None,
        temperature=0,
        **llama_cpp_kwargs,
    ):
        self.sanitize_prompt(prompt=prompt, history=history, schema=schema)

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
                "content": [{"type": "text", "text": prompt}],
            },
        )

        kwargs = {}
        if schema:
            grammar = to_grammar(schema)
            kwargs = {"grammar": grammar}

        # Allow to call `ask` and free up memory immediately
        model = getattr(self, "model", None)

        if model:
            completion = model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=-1,
                **kwargs,
            )
        else:
            with self as wrapper:
                completion = wrapper.model.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    **kwargs,
                )

        return ChatCompletion.parse(completion)
