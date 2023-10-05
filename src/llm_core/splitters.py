# -*- coding: utf-8 -*-
import itertools
import codecs

from dataclasses import dataclass


@dataclass
class TokenSplitter:
    model: str = "gpt-3.5-turbo"
    chunk_size: int = 300
    chunk_overlap: int = 0

    def __post_init__(self):
        if self.chunk_overlap > self.chunk_size:
            raise IndexError(
                f"Overlap {self.chunk_overlap} > window size {self.chunk_size}"
            )

    def first_extract(self, text: str):
        return next(self.chunkify(text))

    def compute_token_count(self, text: str):
        return len(codecs.encode(text, self.model))

    def chunkify(self, text: str):
        tokens = codecs.encode(text, self.model)

        start = 0
        length = len(tokens)

        while True:
            stop = start + self.chunk_size
            chunk = list(itertools.islice(tokens, start, stop))
            yield codecs.decode(chunk, self.model)

            if stop >= length:
                break

            start += self.chunk_size - self.chunk_overlap
