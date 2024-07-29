# -*- coding: utf-8 -*-
import traceback
import platform
import llama_cpp
import dirtyjson

from pathlib import Path
from dataclasses import dataclass

from .base import (
    LLMBase,
    ChatCompletion,
)
from ..schema import (
    to_grammar,
    from_dict,
    make_selection_tool,
)
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
        model_path = str(Path(MODELS_CACHE_DIR) / self.name)
        llama_cpp_kwargs = {
            "n_ctx": self.ctx_size,
            "verbose": self.verbose,
        }

        if platform.system() == "Darwin" and platform.machine() == "arm64":
            # Offload everything onto the GPU on MacOS
            llama_cpp_kwargs["n_gpu_layers"] = 100
            llama_cpp_kwargs["n_threads"] = 1

        if self.llama_cpp_kwargs:
            llama_cpp_kwargs.update(self.llama_cpp_kwargs)

        self.model = llama_cpp.Llama(model_path, **llama_cpp_kwargs)

    def _generate_completion(self, messages, temperature, grammar=None):
        # Allow to call `ask` and free up memory immediately
        model = getattr(self, "model", None)

        if model:
            completion = model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                grammar=grammar,
            )
        else:
            with self as on_demand_wrapper:
                completion = on_demand_wrapper.model.create_chat_completion(
                    messages=messages,
                    temperature=temperature,
                    grammar=grammar,
                )

        return ChatCompletion.parse(completion)

    def ask(self, prompt, history=(), schema=None, temperature=0, tools=None):
        self.sanitize_prompt(prompt=prompt, history=history, schema=schema)

        grammar = None
        if schema:
            grammar = to_grammar(schema)

        if tools:
            tool_selector = make_selection_tool(tools)

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

            completion = self._generate_completion(
                messages, temperature, tool_selector.grammar
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

            return self._generate_completion(messages, temperature, grammar)

        messages = [
            {
                "role": "system",
                "content": self.system_prompt,
            },
        ]
        messages.extend(history)

        messages.append(
            {
                "role": "user",
                "content": prompt,
            },
        )

        return self._generate_completion(messages, temperature, grammar)


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

        model_path = str(Path(MODELS_CACHE_DIR) / self.name)
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
