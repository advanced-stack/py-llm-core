# -*- coding: utf-8 -*-
import platform
import llama_cpp

from pathlib import Path
from dataclasses import dataclass
from typing import Callable

from .base import LLMBase
from ..schema import to_grammar
from ..settings import MODELS_CACHE_DIR


def load_llama_cpp(llm, **kwargs):
    path = str(Path(MODELS_CACHE_DIR) / llm.name)
    defaults = {"verbose": False, "n_ctx": 0}

    if platform.system() == "Darwin" and platform.machine() == "arm64":
        # Offload everything onto the GPU on MacOS
        defaults["n_gpu_layers"] = 100
        defaults["n_threads"] = 1

    llama_kwargs = {}
    llama_kwargs.update(defaults)
    llama_kwargs.update(kwargs)

    return llama_cpp.Llama(path, **llama_kwargs)


def create_llama_cpp_completion(
    llm,
    model,
    messages,
    temperature,
    tools=None,
    tool_choice=None,
    schema=None,
):
    grammar = None
    if schema:
        grammar = to_grammar(schema)

    completion = llm._client.create_chat_completion(
        messages=messages,
        temperature=temperature,
        grammar=grammar,
        max_tokens=-1,
    )

    return completion


@dataclass
class OpenWeightsModel(LLMBase):
    create_completion: Callable = create_llama_cpp_completion
    loader: Callable = load_llama_cpp
    loader_kwargs: dict = None

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_model()

    def load_model(self):
        kwargs = self.loader_kwargs or {}
        self._client = self.loader(llm=self, **kwargs)
        self.ctx_size = int(self._client.metadata["llama.context_length"])

    def release_model(self):
        del self._client
        self.ctx_size = 0
