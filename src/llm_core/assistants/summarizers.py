# -*- coding: utf-8 -*-
from typing import List
from dataclasses import dataclass
from ..splitters import TokenSplitter


@dataclass
class Summary:
    content: str


@dataclass
class SimpleSummary(Summary):
    system_prompt = """
    You are an expert at extracting key information and summarizing.
    """

    prompt = """Content:
    ```
    {content}
    ```
    --
    Summarize the content in {word_count}.
    """


@dataclass
class DenseSummary(Summary):
    missing_entities: List[str]


@dataclass
class DenserSummaryCollection:
    system_prompt = """
    You are an expert in writing rich and dense summaries in broad domains.
    """

    prompt = """Content:
    ```
    {content}
    ```
    --

    You will generate increasingly concise, entity-dense summaries of the above
    Article.

    Repeat the following 2 steps 5 times.

    - Step 1: Identify 1-3 informative Entities from the Article
    which are missing from the previously generated summary and are the most
    relevant.

    - Step 2: Write a new, denser summary of identical length which covers
    every entity and detail from the previous summary plus the missing entities

    A Missing Entity is:

    - Relevant: to the main story
    - Specific: descriptive yet concise (5 words or fewer)
    - Novel: not in the previous summary
    - Faithful: present in the Article
    - Anywhere: located anywhere in the Article

    Guidelines:
    - The first summary should be long (4-5 sentences, approx. {word_count} words) yet
    highly non-specific, containing little information beyond the entities
    marked as missing.

    - Use overly verbose language and fillers (e.g. "this article discusses") to
    reach approx. {word_count} words.

    - Make every word count: re-write the previous summary to improve flow and
    make space for additional entities.

    - Make space with fusion, compression, and removal of uninformative phrases
    like "the article discusses"

    - The summaries should become highly dense and concise yet self-contained,
    e.g., easily understood without the Article.

    - Missing entities can appear anywhere in the new summary.

    - Never drop entities from the previous summary. If space cannot be made,
    add fewer new entities.

    > Remember to use the exact same number of words for each summary.
    Answer in JSON.

    > The JSON in `summaries` should be a list (length 5) of
    dictionaries whose keys are "missing_entities" and "content".

    """

    summaries: List[DenseSummary]


@dataclass
class Summarizer:
    model: str
    assistant_cls: type
    results_cls: type = SimpleSummary

    def fast_summarize(self, content, word_count=80):
        with self.assistant_cls(
            self.results_cls, model=self.model
        ) as assistant:
            splitter = TokenSplitter(
                model=assistant.model_name,
                chunk_size=int(assistant.ctx_size * 0.6),
                chunk_overlap=int(assistant.ctx_size * 0.05),
            )
            chunk = next(splitter.chunkify(content))
            summary = assistant.process(content=chunk, word_count=word_count)
            return summary

    def summarize(self, content, word_count=80):
        with self.assistant_cls(
            self.results_cls, model=self.model
        ) as assistant:
            splitter = TokenSplitter(
                model=assistant.model_name,
                chunk_size=int(assistant.ctx_size * 0.6),
                chunk_overlap=int(assistant.ctx_size * 0.05),
            )
            for chunk in splitter.chunkify(content):
                summary = assistant.process(
                    content=chunk, word_count=word_count
                )
                yield summary
