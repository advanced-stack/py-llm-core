# -*- coding: utf-8 -*-
from pathlib import Path
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
class NuExtractModel(LLMBase):
    """
    Model wrapper using the https://github.com/ggerganov/llama.cpp library

    Not limited to LLaMA models. For the complete list of supported models,

    see https://github.com/ggerganov/llama.cpp#readme
    """

    name: str = "nuextract-tiny"
    system_prompt: str = "You are a helpful assistant"
    ctx_size: int = 2000
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
                # Force CPU inference as there is an issue with Metal
                # for this model
                self.llama_cpp_kwargs["n_gpu_layers"] = 0
                self.llama_cpp_kwargs["n_threads"] = 4

        model_path = str(Path(MODELS_CACHE_DIR) / self.name)

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

        kwargs = {
            "response_format": {
                "type": "json_object",
            },
        }
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
