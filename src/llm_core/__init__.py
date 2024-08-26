import codecs

from .tokenizers import lookup

codecs.register(lookup)
