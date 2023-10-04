# PyLLMCore

## Overview

PyLLMCore provides a light-weighted structured interface with Large
Language Models API.

Use cases with examples are described :

- Question answering using [Chain of Verification](https://arxiv.org/abs/2309.11495)
- Summarize using the [Chain of Density prompting](https://arxiv.org/abs/2309.04269)
- parse unstructured text and obtain Python objects (a populated dataclass)
- extract information
- describe arbitrary tasks for the LLM to perform (translation, ...)


The latest version supports OpenAI, LLaMA and Mistral AI models.

![](./assets/example-cove.jpg)

## Changelog

- 1.4.0: Free up resources in LLamaParser when exiting the context manager
- 1.3.0: Support for LLaMA based models (llama, llama2, Mistral Instruct)
- 1.2.0: Chain of density prompting implemented with OpenAI
- 1.1.0: Chain of Verification implemented with OpenAI
- 1.0.0: Initial version

### How to install

```shell
pip install py-llm-core

# Add you OPENAI_API_KEY to the environment
export OPENAI_API_KEY=sk-<replace with your actual api key>

```

## Use cases

## Question answering with Chain of Verification

The following example implements the technique from the paper [Chain-of-Verification Reduces Hallucination in Large Language Models](https://arxiv.org/abs/2309.11495) from Shehzaad Dhuliawala et al. (2023)

```python
>>> from llm_core.assistants import COVQuestionAnswering
>>> cov_qa = COVQuestionAnswering.ask(
...     question="Name some politicians who were born in NY, New York"
... )
>>> print(cov_qa.revised_answer)

Some politicians who were born in NY, New York include Donald Trump,
Franklin D. Roosevelt, Theodore Roosevelt, and Andrew Cuomo.
```

## Summarizing with Chain of Density Prompting

The following example implements the technique from the paper [From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting](https://arxiv.org/abs/2309.04269) from Adams et al. (2023).


```python
>>> from llm_core.assistants import DenserSummaryCollection
>>> import wikipedia
>>> text = wikipedia.page("Foundation from Isaac Asimov").content
>>> summary_collection = DenserSummaryCollection.summarize(text)
>>> print(summary_collection)
```

```python
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

        ...

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


### Parsing

#### Using llama.cpp grammar and bindings

In the example we use a quantized version of the Mistral AI model. You can download the model weights on [Hugging Face](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/)

```python
from dataclasses import dataclass
from llm_core.parsers import LLamaParser

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


with LLamaParser(Book, model_path="mistral-7b-instruct-v0.1.Q4_K_M.gguf") as parser:
    book = parser.parse(text)
    print(book)
```

```python
Book(
    title='Foundation',
    summary="""Foundation is a science fiction novel by American writer
        Isaac Asimov. It is the first published in his Foundation Trilogy
        (later expanded into the Foundation series). Foundation is a
        cycle of five interrelated short stories, first published as a
        single book by Gnome Press in 1951. Collectively they tell the
        early story of the Foundation, an institute founded by 
        psychohistorian Hari Seldon to preserve the best of galactic
        civilization after the collapse of the Galactic Empire.""",
    author='Isaac Asimov',
    published_year=1951
)
```

#### Using OpenAI functions

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
```

```python
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
import wikipedia
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

```

```python
BookCollection(
    books=[
        Book(
            title="Foundation",
            summary="The first book in the Foundation series. It introduces the concept of psychohistory and follows the mathematician Hari Seldon as he predicts the fall of the Galactic Empire and establishes the Foundation to preserve knowledge and shorten the Dark Age.",
            author="Isaac Asimov",
            published_year=1951,
        ),
        ...
        
        Book(
            title="Forward the Foundation",
            summary="The final book in the Foundation series. It takes place eight years after Prelude to Foundation and explores Hari Seldon's final years as he works to establish the Second Foundation.",
            author="Isaac Asimov",
            published_year=1993,
        ),
    ]
)
```




## Performing arbitrary tasks (summary, translations,...)

### Example : Summary + fact extraction

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
```

```python

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

