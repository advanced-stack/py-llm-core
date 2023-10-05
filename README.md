# PyLLMCore

## Overview

PyLLMCore is a light-weighted structured interface with Large Language Models 
with native support for [llama.cpp](http://github.com/ggerganov/llama.cpp) and OpenAI API.

The design decisions behind PyLLMCore are:

- Pythonic internal API
- Sane defaults all the way
- Clear abstractions and terminology
- Out of the box utility classes

## Models supported

Besides OpenAI API, the following models are supported for local inference using the [llama.cpp](http://github.com/ggerganov/llama.cpp):

- LLaMA
- LLaMA 2
- Falcon
- [Alpaca](https://github.com/ggerganov/llama.cpp#instruction-mode-with-alpaca)
- [GPT4All](https://github.com/ggerganov/llama.cpp#using-gpt4all)
- [Chinese LLaMA / Alpaca](https://github.com/ymcui/Chinese-LLaMA-Alpaca) and [Chinese LLaMA-2 / Alpaca-2](https://github.com/ymcui/Chinese-LLaMA-Alpaca-2)
- [Vigogne (French)](https://github.com/bofenghuang/vigogne)
- [Vicuna](https://github.com/ggerganov/llama.cpp/discussions/643#discussioncomment-5533894)
- [Koala](https://bair.berkeley.edu/blog/2023/04/03/koala/)
- [OpenBuddy 🐶 (Multilingual)](https://github.com/OpenBuddy/OpenBuddy)
- [Pygmalion 7B / Metharme 7B](#using-pygmalion-7b--metharme-7b)
- [WizardLM](https://github.com/nlpxucan/WizardLM)
- [Baichuan-7B](https://huggingface.co/baichuan-inc/baichuan-7B) and its derivations (such as [baichuan-7b-sft](https://huggingface.co/hiyouga/baichuan-7b-sft))
- [Aquila-7B](https://huggingface.co/BAAI/Aquila-7B) / [AquilaChat-7B](https://huggingface.co/BAAI/AquilaChat-7B)
- [Starcoder models](https://github.com/ggerganov/llama.cpp/pull/3187)
- [Mistral AI v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)

## Use cases

PyLLMCore covers a narrow range of use cases and serves as a building brick:

- Parsing: see the `parsers` module
- Summarizing: see the `assistants.summarizers` module
- Question answering: see the `assistants.analyst` module
- Hallucinations reduction: see the `assistants.verifiers` module
- Context size management: see the `splitters` module
- Tokenizing, encoding, decoding: see the `token_codecs` module


## Changelog

- 1.5.0: 
    + Refactoring
    + Renamed `LLamaParser` into `LLaMACPPParser`
    + Dynamically enable GPU offloading on MacOS
    + Added configuration option for storing local models

- 1.4.0: Free up resources in LLamaParser when exiting the context manager
- 1.3.0: Support for LLaMA based models (llama, llama2, Mistral Instruct)
- 1.2.0: Chain of density prompting implemented with OpenAI
- 1.1.0: Chain of Verification implemented with OpenAI
- 1.0.0: Initial version

## Install

### Quick start

```shell
pip install py-llm-core

# Add you OPENAI_API_KEY to the environment
export OPENAI_API_KEY=sk-<replace with your actual api key>

# For local inference with GGUF models, store your models in MODELS_CACHE_DIR
mkdir -p ~/.cache/py-llm-core/models
cd ~/.cache/py-llm-core/models
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf

```

### Troubleshooting

The `llama-cpp-python` dependency may improperly detects the architecture and raise an error `an incompatible architecture (have 'x86_64', need 'arm64'))`.

If that's the case, run the following in your virtual env:

```shell
CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64" pip3 install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python
```


## Documentation


### Parsing

#### Use Parser classes

Available parsers:

- `parsers.OpenAIParser`
- `parsers.LLaMACPPParser`

#### Using a local model : Mistral AI Instruct

```python
from dataclasses import dataclass
from llm_core.parsers import LLaMACPPParser

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


with LLaMACPPParser(Book, model="mistral-7b-instruct-v0.1.Q4_K_M.gguf") as parser:
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

#### Using OpenAI

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


### Perform tasks

To perform generic tasks, you will use the `assistants` module that provides generic assistants:

- `assistants.OpenAIAssistant`
- `assistants.LLaMACPPAssistant`


#### Summarizing

```python
import wikipedia
from llm_core.assistants import Summarizer, LLaMACPPAssistant


summarizer = Summarizer(
    model="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    assistant_cls=LLaMACPPAssistant
)

text = wikipedia.page("Foundation from Isaac Asimov").content

# To summarize only with 50% of the model context size
partial_summary = summarizer.fast_summarize(text)

# Iterative summaries on the whole content
for summary in summarizer.summarize(text):
    print(summary)

```

#### Reduce hallucinations using verifiers

This example implements loosely the Chain of Verification (CoVe).

To reduce hallucinations in the LLM completions, you can use the following example
as a starting point:

```python
import requests
from llm_core.assistants import Analyst, Doubter, ConsistencyVerifier, LLaMACPPAssistant


context = requests.get("https://raw.githubusercontent.com/hendricius/pizza-dough/main/README.md").text

# Analyst answer questions
analyst = Analyst(
    model="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    assistant_cls=LLaMACPPAssistant,
)

instructions = "Write a recipe to make a margherita pizza"

analyst_response = analyst.ask(instructions, context)

# Doubter provides with verification questions when a previous completion
# has been done
doubter = Doubter(
    model="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    assistant_cls=LLaMACPPAssistant,
)

question_collection = doubter.verify(
    '\n'.join((context, instructions)),
    analyst_response.content,
    n_questions=5
)
print(question_collection)
```

```python
QuestionCollection(
    questions=[
        'Is the weight of the flour specified in grams or percentages?',
        'What is the recommended temperature for the oven?',
        'How long should the dough be left to rest before shaping it into a pizza?',
        'What type of yeast should be used (dry or fresh)?',
        'Do you need to proof the dough before using it to make a pizza?'
    ]
)
```

Then answer individual questions with the Analyst:

```python
# You can even segregate models between analyst vs verifier to improve reliability
verifier = ConsistencyVerifier(
    model="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    assistant_cls=LLaMACPPAssistant,
)

template = """Q:{}
A:{}
Consistent:{}"""

for response in analyst.batch_ask(question_collection.questions, context=context):
    answer_consistency = verifier.verify(
        question=question,
        context=context,
        answer=response.content
    )
    print(template.format(question, response.content, answer_consistency.is_consistent))

```

From there, you can revise the overall answer.

#### Using the assistants module

The following example using the `assistants.analysts` module shows how to use assistants to generate a simple recommendation.

```python
from dataclasses import dataclass
from llm_core.assistants import Analyst, Answer, LLaMACPPAssistant

context = """
Foundation is a science fiction novel by American writer
Isaac Asimov. It is the first published in his Foundation Trilogy (later
expanded into the Foundation series). Foundation is a cycle of five
interrelated short stories, first published as a single book by Gnome Press
in 1951. Collectively they tell the early story of the Foundation,
an institute founded by psychohistorian Hari Seldon to preserve the best
of galactic civilization after the collapse of the Galactic Empire.
----
The user likes the movie Interstellar
"""

@dataclass
class Recommendation(Answer):
    is_recommended: bool


analyst = Analyst(
    model="mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    assistant_cls=LLaMACPPAssistant,
    results_cls=Recommendation,
)

response = analyst.ask("Should we recommend Foundation ?", context=context)
print(response)
```

```python
Recommendation(
    content='Foundation is a science fiction novel by Isaac Asimov that tells the early story of the Foundation, an institute founded by psychohistorian Hari Seldon to preserve the best of galactic civilization after the collapse of the Galactic Empire. The user has not mentioned any specific reasons for liking or disliking the movie Interstellar, so it is difficult to determine if they would also enjoy Foundation. However, if the user enjoys science fiction and exploring complex ideas about the future of humanity, then Foundation may be a good recommendation.',
    is_recommended=True
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


splitter = TokenSplitter(model="mistral-7b-instruct-v0.1.Q4_K_M.gguf", chunk_size=50, chunk_overlap=0)

for chunk in splitter.chunkify(text):
    print(chunk)

```
