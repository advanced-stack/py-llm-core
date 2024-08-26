# -*- coding: utf-8 -*-
import platform
import llama_cpp

from pathlib import Path
from dataclasses import dataclass
from typing import Callable

from .base import LLMBase
from ..schema import to_grammar
from ..settings import MODELS_CACHE_DIR


def load_model(llm, **kwargs):
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


def create_completion(
    llm,
    model,
    messages,
    temperature,
    tools=None,
    tool_choice=None,
    schema=None,
):
    # Allow to call `ask` and free up memory immediately
    model = getattr(llm, "_model", None)

    grammar = None
    if schema:
        grammar = to_grammar(schema)

    if model:
        completion = model.create_chat_completion(
            messages=messages,
            temperature=temperature,
            grammar=grammar,
            max_tokens=-1,
        )
    else:
        with llm as on_demand_wrapper:
            completion = on_demand_wrapper.model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                grammar=grammar,
                max_tokens=-1,
            )

    return completion


@dataclass
class OpenWeightsModel(LLMBase):
    create_completion: Callable = create_completion
    model_loader: Callable = load_model
    model_loader_kwargs: dict = None

    def __enter__(self):
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release_model()

    def load_model(self):
        kwargs = self.model_loader_kwargs or {}
        self._model = self.model_loader(llm=self, **kwargs)
        self.ctx_size = int(self._model.metadata["llama.context_length"])

    def release_model(self):
        del self._model
        self.ctx_size = 0
