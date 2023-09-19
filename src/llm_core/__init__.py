import codecs

from .tokenizers import tiktoken_lookup

codecs.register(tiktoken_lookup)
