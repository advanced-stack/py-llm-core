import codecs

from .token_codecs import lookup

codecs.register(lookup)
