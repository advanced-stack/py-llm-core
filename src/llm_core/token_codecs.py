# -*- coding: utf-8 -*-
import codecs
import tiktoken
import llama_cpp
import os

from .llm import LLaMACPPModel

from .settings import MODELS_CACHE_DIR

codec_to_model_name = {}
for model_name in os.listdir(MODELS_CACHE_DIR):
    key = model_name.replace("-", "_").lower()
    codec_to_model_name[key] = model_name


class BaseTokenCodec:
    def encode(self, input, errors="strict"):
        output = self.token_encoder.encode(input)
        return output, len(output)

    def decode(self, input, errors="strict"):
        output = self.token_encoder.decode(input)
        return output, len(output)


class OpenAIGPTCodec(BaseTokenCodec):
    @classmethod
    def is_supported(cls, codec_name):
        return codec_name in (
            "gpt3.5",
            "gpt3.5_0613",
            "gpt_3.5",
            "gpt_3.5_0613",
            "gpt_3.5_turbo",
            "gpt_3.5_turbo_0613",
            "gpt_3.5_turbo_16k",
            "gpt_3.5_turbo_16k_0613",
            "gpt_4",
            "gpt_4_0613",
            "gpt_4_1106_preview",
            "gpt_4_32k",
        )

    def __init__(self, codec_name):
        # Map everything to gpt-3.5-turbo
        self.token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")


class LLaMACodec:
    @classmethod
    def is_supported(cls, codec_name):
        return codec_name in codec_to_model_name

    def __init__(self, codec_name):
        self.model_name = codec_to_model_name[codec_name]

    def encode(self, input, errors="strict"):
        with LLaMACPPModel(name=self.model_name) as wrapper:
            tokenizer = llama_cpp.LlamaTokenizer(wrapper.model)
            output = tokenizer.encode(input)

        return output, len(output)

    def decode(self, input, errors="strict"):
        with LLaMACPPModel(name=self.model_name) as wrapper:
            tokenizer = llama_cpp.LlamaTokenizer(wrapper.model)
            output = tokenizer.decode(input)

        return output, len(output)


def lookup(codec_name):
    if OpenAIGPTCodec.is_supported(codec_name):
        codec = OpenAIGPTCodec(codec_name)

    elif LLaMACodec.is_supported(codec_name):
        codec = LLaMACodec(codec_name)

    else:
        return None

    return codecs.CodecInfo(
        name=codec_name,
        encode=codec.encode,
        decode=codec.decode,
    )
