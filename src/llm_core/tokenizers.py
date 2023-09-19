# -*- coding: utf-8 -*-
import codecs
import tiktoken


class TikTokenCodec:
    def __init__(self, token_encoder):
        self.token_encoder = token_encoder

    def encode(self, input, errors="strict"):
        output = self.token_encoder.encode(input)
        return output, len(output)

    def decode(self, input, errors="strict"):
        output = self.token_encoder.decode(input)
        return output, len(output)


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
