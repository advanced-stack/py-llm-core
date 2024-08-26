# -*- coding: utf-8 -*-
import codecs
import tiktoken


class TiktokenCodec:
    """
    Tiktoken is used to estimate the number of tokens of a given string
    """

    @classmethod
    def is_supported(cls, codec_name):
        return codec_name == "tiktoken"

    def __init__(self, codec_name):
        # Map everything to gpt-3.5-turbo
        self.token_encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")

    def encode(self, input, errors="strict"):
        output = self.token_encoder.encode(input)
        return output, len(output)

    def decode(self, input, errors="strict"):
        output = self.token_encoder.decode(input)
        return output, len(output)


def lookup(codec_name):
    if TiktokenCodec.is_supported(codec_name):
        codec = TiktokenCodec(codec_name)
    else:
        return None

    return codecs.CodecInfo(
        name=codec_name,
        encode=codec.encode,
        decode=codec.decode,
    )
