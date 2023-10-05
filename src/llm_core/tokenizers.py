# -*- coding: utf-8 -*-
import codecs
import tiktoken
import llama_cpp


class BaseTokenCodec:
    def encode(self, input, errors="strict"):
        output = self.token_encoder.encode(input)
        return output, len(output)

    def decode(self, input, errors="strict"):
        output = self.token_encoder.decode(input)
        return output, len(output)


class OpenAIGPTCodec(BaseTokenCodec):
    codec_names = set(
        (
            "gpt3.5",
            "gpt_3.5",
            "gpt_3.5_turbo",
            "gpt_3.5_turbo_16k",
            "gpt_4",
            "gpt_4_32k",
        )
    )

    def __init__(self):
        self.token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")


class LLaMACodec(BaseTokenCodec):
    codec_names = set(
        (
            "mistral_7b",
            "llama",
            "llama2",
        )
    )

    def __init__(self):
        self.token_encoder = llama_cpp.LlamaTokenizer()


def tiktoken_lookup(codec_name):
    original_model_name = model_name.replace("_", "-")
    try:
        encoder = tiktoken.encoding_for_model(original_model_name)
        codec = TikTokenCodec(encoder)
    except KeyError:
        # Explicit pass so other codecs can answer
        return None

    return codecs.CodecInfo(
        name=codec_name,
        encode=codec.encode,
        decode=codec.decode,
    )


def tiktoken_lookup(model_name):
    original_model_name = model_name.replace("_", "-")
    try:
        encoder = tiktoken.encoding_for_model(original_model_name)
        codec = TikTokenCodec(encoder)
    except KeyError:
        # Explicit pass so other codecs can answer
        return None

    return codecs.CodecInfo(
        name=model_name,
        encode=codec.encode,
        decode=codec.decode,
    )
