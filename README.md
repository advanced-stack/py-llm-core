# PyLLMCore

## Overview

PyLLMCore provides a light-weighted structured interface with Large
Language Models API.

The current version supports OpenAI API through the `openai` package.


### What we do with this library ?

Use cases with examples are described hereunder, but basically you can :

- parse unstructured text and obtain Python objects (a populated dataclass)
- describe tasks for the LLM to perform (summarization, translation, ...)

### How to install

```shell
pip install py-llm-core

# Add you OPENAI_API_KEY to the environment
export OPENAI_API_KEY=sk-<replace with your actual api key>

```


## Use cases

### Parsing

When given unstructured content, LLMs are powerful enough to extract information and produce structured content.

We use a dataclass as a light-weighted structure to hold parsed data.

```python
from dataclasses import dataclass
from llm_core.parsers import OpenAIParser


@dataclass
class Book:
    title: str
    summary: str
    author: str
    published_year: int


text = """Foundation is a science fiction novel by American writer
Isaac Asimov. It is the first published in his Foundation Trilogy (later
expanded into the Foundation series). Foundation is a cycle of five
interrelated short stories, first published as a single book by Gnome Press
in 1951. Collectively they tell the early story of the Foundation,
an institute founded by psychohistorian Hari Seldon to preserve the best
of galactic civilization after the collapse of the Galactic Empire.
"""


with OpenAIParser(Book) as parser:
    book = parser.parse(text)
    print(book)

# The results are :

Book(
    title='Foundation',
    summary="""Foundation is a cycle of five interrelated
    short stories, first published as a single book by Gnome Press in 1951.
    Collectively they tell the early story of the Foundation, an institute
    founded by psychohistorian Hari Seldon to preserve the best of galactic
    civilization after the collapse of the Galactic Empire.""",
    author='Isaac Asimov',
    published_year=1951
)
```

### Summary and advanced information extraction

We can use all the abilities to perform all kind of text processing with the
same class.

```python
from typing import List
import wikipedia  # Run `make test-setup` to install wikipedia package
from dataclasses import dataclass
from llm_core.parsers import OpenAIParser


@dataclass
class Book:
    title: str
    summary: str
    author: str
    published_year: int


@dataclass
class BookCollection:
    books: List[Book]


text = wikipedia.page("Foundation from Isaac Asimov").content


with OpenAIParser(BookCollection, model='gpt-3.5-turbo-16k') as parser:
    book_collection = parser.parse(text)
    print(book_collection)


# The results are :


BookCollection(
    books=[
        Book(
            title="Foundation",
            summary="The first book in the Foundation series. It introduces the concept of psychohistory and follows the mathematician Hari Seldon as he predicts the fall of the Galactic Empire and establishes the Foundation to preserve knowledge and shorten the Dark Age.",
            author="Isaac Asimov",
            published_year=1951,
        ),
        Book(
            title="Foundation and Empire",
            summary="The second book in the Foundation series. It explores the conflict between the Foundation and the declining Galactic Empire, as well as the rise of a powerful mutant known as the Mule.",
            author="Isaac Asimov",
            published_year=1952,
        ),
        Book(
            title="Second Foundation",
            summary="The third book in the Foundation series. It reveals the existence of the Second Foundation, a secret organization with psychic abilities, and their efforts to guide the course of history.",
            author="Isaac Asimov",
            published_year=1953,
        ),
        Book(
            title="Foundation's Edge",
            summary="The fourth book in the Foundation series. It takes place 500 years after the establishment of the Foundation and follows Golan Trevize as he searches for the mythical planet Earth.",
            author="Isaac Asimov",
            published_year=1982,
        ),
        Book(
            title="Foundation and Earth",
            summary="The fifth book in the Foundation series. It continues the search for Earth and explores the merging of R. Daneel Olivaw's mind with a human child's intellect.",
            author="Isaac Asimov",
            published_year=1986,
        ),
        Book(
            title="Prelude to Foundation",
            summary="A prequel to the Foundation series. It follows Hari Seldon as he develops the theory of psychohistory and faces various challenges on the planet Trantor.",
            author="Isaac Asimov",
            published_year=1988,
        ),
        Book(
            title="Forward the Foundation",
            summary="The final book in the Foundation series. It takes place eight years after Prelude to Foundation and explores Hari Seldon's final years as he works to establish the Second Foundation.",
            author="Isaac Asimov",
            published_year=1993,
        ),
    ]
)
```


## Tokenizer

Tiktoken library is registered as a codec within the Python
codecs registry :

```python
import llm_core
import codecs

text = """Foundation is a science fiction novel by American writer
Isaac Asimov. It is the first published in his Foundation Trilogy (later
expanded into the Foundation series). Foundation is a cycle of five
interrelated short stories, first published as a single book by Gnome Press
in 1951. Collectively they tell the early story of the Foundation,
an institute founded by psychohistorian Hari Seldon to preserve the best
of galactic civilization after the collapse of the Galactic Empire.
"""


# You can encode the text into tokens like that:

tokens = codecs.encode(text, 'gpt-3.5-turbo')

print(tokens)
[19137, 374, 264, 8198, ... 627]


print(len(tokens))

100

```

## Performing tasks (summary, translations,...)

When a task should be performed by the language model, 
we add an explicit prompt (and system_prompt) to the desired structure.


```python
from typing import List
from dataclasses import dataclass
from llm_core.assistants import OpenAIAssistant


@dataclass
class SummaryWithInsights:
    system_prompt = """
    You are a world-class copy writer and work in broad domains.
    You help users produce better analysis of content by summarizing
     written content.
    """

    prompt = """
    Article:

    {content}

    - Summarize the previous content in approx. {word_count} words.
    - Provide a list key facts

    """

    summary: str
    facts: List[str]

    @classmethod
    def summarize(cls, content, word_count=100):
        with OpenAIAssistant(cls, model='gpt-3.5-turbo-16k') as assistant:
            return assistant.process(content=content, word_count=word_count)




import wikipedia  # Run `make test-setup` to install wikipedia package
text = wikipedia.page("Foundation from Isaac Asimov").content

response = SummaryWithInsights.summarize(text)

print(response)

# prints the following

SummaryWithInsights(
    summary="""The Foundation series is a science fiction book series written 
    by Isaac Asimov. It was first published as a series of short stories and
     novellas in 1942–50 and subsequently in three collections in 1951–53. The 
     series follows the mathematician Hari Seldon as he develops a theory of
      psychohistory, a new mathematics of sociology that can predict the future
       of large populations. The series explores the rise and fall of a future
        Galactic Empire and the efforts of the Foundation to preserve 
        civilization during a Dark Age. The series has had a significant
         cultural impact and has won several awards.""",
    facts=[
        "The Foundation series was written by Isaac Asimov.",
        "It was first published as a series of short stories and novellas in 1942–50.",
        "The series follows the mathematician Hari Seldon and his development of psychohistory.",
        "The series explores the rise and fall of a future Galactic Empire.",
        "The Foundation works to preserve civilization during a Dark Age.",
        "The series has had a significant cultural impact and has won several awards.",
    ],
)


```


## Chunking and splitting

```python
from llm_core.splitters import TokenSplitter


text = """Foundation is a science fiction novel by American writer
Isaac Asimov. It is the first published in his Foundation Trilogy (later
expanded into the Foundation series). Foundation is a cycle of five
interrelated short stories, first published as a single book by Gnome Press
in 1951. Collectively they tell the early story of the Foundation,
an institute founded by psychohistorian Hari Seldon to preserve the best
of galactic civilization after the collapse of the Galactic Empire.
"""


splitter = TokenSplitter(chunk_size=50, chunk_overlap=0)

for chunk in splitter.chunkify(text):
    print(chunk)


```

## Summarizing with Chain of Density Prompting

The following example implements the technique from the paper 
"From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting", Adams et al. (2023).

```python
from typing import List
from dataclasses import dataclass
from llm_core.assistants import OpenAIAssistant


@dataclass
class DenseSummary:
    denser_summary: str
    missing_entities: List[str]


@dataclass
class DenserSummaryCollection:
    system_prompt = """
    You are an expert in writing rich and dense summaries in broad domains.
    """

    prompt = """
    Article:
    
    {article}

    ----

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
    - The first summary should be long (4-5 sentences, approx. 80 words) yet
    highly non-specific, containing little information beyond the entities
    marked as missing.

    - Use overly verbose language and fillers (e.g. "this article discusses") to
    reach approx. 80 words.

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

    > The JSON in `summaries_per_step` should be a list (length 5) of
    dictionaries whose keys are "missing_entities" and "denser_summary".

    """

    summaries: List[DenseSummary]


    @classmethod
    def summarize(cls, article):
        with OpenAIAssistant(cls, model='gpt-4') as assistant:
            return assistant.process(article=article)



import wikipedia  # Run `make test-setup` to install wikipedia package
text = wikipedia.page("Foundation from Isaac Asimov").content

response = DenserSummaryCollection.summarize(text)
print(response)


DenserSummaryCollection(
    summaries=[
        DenseSummary(
            denser_summary="""This article discusses the Foundation series, a 
            science fiction book series written by American author Isaac Asimov.
            The series was first published as a series of short stories and
            novellas in 1942–50, and subsequently in three collections in
            1951–53. The premise of the stories is that in the waning days of
            a future Galactic Empire, the mathematician Hari Seldon spends his
            life developing a theory of psychohistory, a new and effective 
            mathematics of sociology.""",
            missing_entities=["Isaac Asimov", "Hari Seldon", "psychohistory"],
        ),
        DenseSummary(
            denser_summary="""Isaac Asimov's Foundation series, initially a
            trilogy, explores the concept of psychohistory, a mathematical
            sociology developed by Hari Seldon. In the waning days of a future
            Galactic Empire, Seldon predicts its fall and a subsequent
            Dark Age. He devises a plan to limit this interregnum to a thousand
            years, rather than the foreseen 30,000. The series was later
            expanded with two sequels and two prequels.""",
            missing_entities=["Galactic Empire", "Dark Age", "interregnum"],
        ),
        DenseSummary(
            denser_summary="""Isaac Asimov's Foundation series, set in a future
            Galactic Empire's decline, introduces Hari Seldon's psychohistory,
            predicting the Empire's fall and a 30,000-year Dark Age. Seldon's
            plan aims to limit this interregnum to a thousand years. The
            series, initially a trilogy, was expanded with two sequels and two
            prequels. The original stories were inspired by Edward Gibbon's
            History of the Decline and Fall of the Roman Empire.""",
            missing_entities=[
                "Edward Gibbon's History of the Decline and Fall of the Roman Empire"
            ],
        ),
        DenseSummary(
            denser_summary="""Isaac Asimov's Foundation series, inspired by
            Edward Gibbon's History of the Decline and Fall of the Roman
            Empire, explores Hari Seldon's psychohistory predicting the fall of
            a future Galactic Empire and a 30,000-year Dark Age. Seldon's plan
            aims to limit this interregnum to a thousand years. The series,
            initially a trilogy, was expanded with two sequels and two
            prequels. The plot follows the series' in-universe chronology,
            not the order of publication.""",
            missing_entities=["in-universe chronology"],
        ),
        DenseSummary(
            denser_summary="""Isaac Asimov's Foundation series, inspired by
            Edward Gibbon's History of the Decline and Fall of the Roman
            Empire, explores Hari Seldon's psychohistory predicting the fall of
            a future Galactic Empire and a 30,000-year Dark Age. Seldon's plan
            aims to limit this interregnum to a thousand years. The series,
            initially a trilogy, was expanded with two sequels and two
            prequels. The plot follows the series' in-universe chronology, not
            the order of publication, and won the one-time Hugo Award for
            'Best All-Time Series' in 1966.""",
            missing_entities=["Hugo Award for 'Best All-Time Series'"],
        ),
    ]
)


```




