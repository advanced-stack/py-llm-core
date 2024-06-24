# -*- coding: utf-8 -*-
import json
import dirtyjson
from dataclasses import asdict

from .llm import OpenAIChatModel, LLaMACPPModel, MistralAILarge, NuExtractModel
from .schema import to_json_schema, from_dict


class BaseParser:
    def __init__(self, target_cls, *args, **kwargs):
        self.target_cls = target_cls
        self.target_json_schema = to_json_schema(self.target_cls)

    def deserialize(self, json_str):
        attributes = dirtyjson.loads(json_str)
        return from_dict(self.target_cls, attributes)

    def parse(self, text):
        prompt = "Extract and parse information from provided content"
        history = [{"role": "user", "content": text}]
        completion = self.model_wrapper.ask(
            prompt, history, schema=self.target_json_schema
        )
        instance = self.deserialize(completion.choices[0].message.content)
        return instance

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class OpenAIParser(BaseParser):
    def __init__(
        self,
        target_cls,
        model="gpt-3.5-turbo",
        completion_kwargs=None,
        *args,
        **kwargs
    ):
        super().__init__(target_cls, *args, **kwargs)
        self.completion_kwargs = (
            {} if completion_kwargs is None else completion_kwargs
        )

        self.model_wrapper = OpenAIChatModel(
            name=model,
            system_prompt=(
                "Act as a powerful AI able to extract, parse and process "
                "information from unstructured content."
            ),
        )
        self.ctx_size = self.model_wrapper.ctx_size
        self.model_name = self.model_wrapper.name


class LLaMACPPParser(BaseParser):
    """
    Parser using the https://github.com/ggerganov/llama.cpp library

    Not limited to LLaMA models. For the complete list of supported models,

    see https://github.com/ggerganov/llama.cpp#readme
    """

    def __init__(
        self,
        target_cls,
        model="mistral",
        completion_kwargs=None,
        llama_cpp_kwargs=None,
        *args,
        **kwargs
    ):
        super().__init__(target_cls, *args, **kwargs)
        self.completion_kwargs = (
            {} if completion_kwargs is None else completion_kwargs
        )

        self.model_wrapper = LLaMACPPModel(
            name=model, llama_cpp_kwargs=llama_cpp_kwargs
        )
        self.ctx_size = self.model_wrapper.ctx_size
        self.model_name = self.model_wrapper.name

    def __enter__(self):
        self.model_wrapper.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model_wrapper.release_model()


class MistralAILargeParser(BaseParser):
    def __init__(
        self,
        target_cls,
        model="mistral-large-latest",
        completion_kwargs=None,
        *args,
        **kwargs
    ):
        super().__init__(target_cls, *args, **kwargs)
        self.completion_kwargs = (
            {} if completion_kwargs is None else completion_kwargs
        )

        self.model_wrapper = MistralAILarge(
            name=model,
            system_prompt=(
                "Act as a powerful AI able to extract, parse and process "
                "information from unstructured content."
            ),
        )
        self.ctx_size = self.model_wrapper.ctx_size
        self.model_name = self.model_wrapper.name


class NuExtractParser(BaseParser):
    """
    Parser using the NuExtract model series
    """

    def __init__(
        self,
        target_cls,
        model="nuextract-tiny",
        completion_kwargs=None,
        llama_cpp_kwargs=None,
        *args,
        **kwargs
    ):
        super().__init__(target_cls, *args, **kwargs)
        self.completion_kwargs = (
            {} if completion_kwargs is None else completion_kwargs
        )

        self.model_wrapper = NuExtractModel(
            name=model, llama_cpp_kwargs=llama_cpp_kwargs
        )
        self.ctx_size = self.model_wrapper.ctx_size
        self.model_name = self.model_wrapper.name

    def __enter__(self):
        self.model_wrapper.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.model_wrapper.release_model()

    def deserialize(self, json_str):
        attributes = dirtyjson.loads(json_str)
        return from_dict(self.target_cls, attributes)

    def parse(self, text):
        simplified_schema = json.dumps(asdict(self.target_cls()), indent=4)
        prompt = """<|input|>
### Template:
{simplified_schema}
### Text:
{text}
<|output|>
""".format(
            simplified_schema=simplified_schema, text=text
        )

        history = []
        completion = self.model_wrapper.ask(prompt, history)
        instance = self.deserialize(completion.choices[0].message.content)
        return instance
