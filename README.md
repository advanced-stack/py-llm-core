# PyLLMCore

## Overview

PyLLMCore is a light-weighted interface with Large Language Models.

It comes with native support:
- OpenAI
- MistralAI
- Anthropic
- a wide range of open-weights models (GGUF) thanks to `llama-cpp-python` bindings

It requires Python 3.8.

## Expected benefits and reasons to use PyLLMCore

- Simple to use and to understand
- Pythonic API
- Easy hacking
- As little dependencies as possible
- Structures are *everywhere* provided by the standard library `dataclasses` module
- Easy swapping between models

## Why you shouldn't use PyLLMCore

- You need a whole framework: Take a look at [langchain](https://github.com/langchain-ai/langchain)
- You need tremendous performance: Take a look at [vllm](https://github.com/vllm-project/vllm)
- You want/need to use Pydantic and don't use the `dataclasses` module

## Use cases

PyLLMCore has evolved to covers a wider range of use cases and serves as a building brick:

- Parsing raw content: see the `parsers` module
- Tool and function calling: see the `assistants` module
- Context window size management: see the `splitters` module


## Install

### Quick start

```shell
pip install py-llm-core

#: To use OpenAI models, set your API key
export OPENAI_API_KEY=sk-<replace with your actual api key>

#: To use local models (i.e. completely offline),
#: download and store your models in ~/.cache/py-llm-core/models/

#: The following commands download the best models (you can use any GGUF models)
#: LLaMA-3.1-8B (Quantized version Q4_K_M)
#: Mistral 7B v0.3 (Quantized version Q4_K_M)

mkdir -p ~/.cache/py-llm-core/models
wget -O ~/.cache/py-llm-core/models/llama-8b-3.1-q4 \
    https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf?download=true

wget -O ~/.cache/py-llm-core/models/mistral-7b-v0.3-q4 \
    https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf?download=true
```

## Documentation

### Parsing

You can use these following examples to extract information from raw text.

```python
from dataclasses import dataclass
from typing import List

@dataclass
class Book:
    title: str
    summary: str
    author: str
    published_year: int

@dataclass
class BookCollection:
    books: List[Book]


text = """The Foundation series is a science fiction book series written by
American author Isaac Asimov. First published as a series of short
stories and novellas in 1942–50, and subsequently in three books in
1951–53, for nearly thirty years the series was widely known as The
Foundation Trilogy: Foundation (1951), Foundation and Empire (1952),
and Second Foundation (1953). It won the one-time Hugo Award for "Best
All-Time Series" in 1966. Asimov later added new volumes, with two
sequels, Foundation's Edge (1982) and Foundation and Earth (1986), and
two prequels, Prelude to Foundation (1988) and Forward the Foundation
(1993).

The premise of the stories is that in the waning days of a future
Galactic Empire, the mathematician Hari Seldon devises the theory of
psychohistory, a new and effective mathematics of sociology. Using
statistical laws of mass action, it can predict the future of large
populations. Seldon foresees the imminent fall of the Empire, which
encompasses the entire Milky Way, and a dark age lasting 30,000 years
before a second empire arises. Although the momentum of the Empire's
fall is too great to stop, Seldon devises a plan by which "the
onrushing mass of events must be deflected just a little" to
eventually limit this interregnum to just one thousand years. The
books describe some of the dramatic events of those years as they are
shaped by the underlying political and social mechanics of Seldon's
Plan.
"""
```

#### Usage with open weights models (gguf)

```python
from llm_core.parsers import OpenWeightsParser

# default model is "mistral-7b-v0.3-q4"
with OpenWeightsParser(BookCollection) as parser:
    books_collection = parser.parse(text)

    for book in books_collection.books:
        print(book)
```

#### Usage with OpenAI models

```python
from llm_core.parsers import OpenAIParser

# default model is "gpt-4o-mini"
with OpenAIParser(BookCollection) as parser:
    books_collection = parser.parse(text)

    for book in books_collection.books:
        print(book)
```

#### Usage with Azure OpenAI deployments

Be sure to add the following environment variables:

- AZURE_OPENAI_API_KEY
- AZURE_OPENAI_ENDPOINT
- AZURE_OPENAI_API_VERSION

```python
# Using Azure
from llm_core.llm import AzureOpenAIChatModel
from llm_core.parsers import OpenAIParser

# default model is "gpt-4o-mini"
with OpenAIParser(BookCollection, model_cls=AzureOpenAIChatModel) as parser:
    books_collection = parser.parse(text)

    for book in books_collection.books:
        print(book)
```

#### Usage with Mistral models

```python
from llm_core.parsers import MistralAIParser

# default model is "open-mistral-nemo"
with MistralAIParser(BookCollection) as parser:
    books_collection = parser.parse(text)

    for book in books_collection.books:
        print(book)
```

### Advanced tasks

#### Using tools a.k.a. Function Calling

We can make the LLM use a tool to enrich its context and produce a better answer.

To use tools, the only thing you need to define is a dataclass and the `__call__`
method and implement the required logic.

Here's an example on how to add web search capabilities using [Brave Search API](https://brave.com/search/api/)

```python
import requests
from decouple import config
from dataclasses import dataclass

from llm_core.llm import OpenAIChatModel


@dataclass
class WebSearchProvider:
    query: str

    def __call__(self):
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": config("BRAVE_AI_API_KEY")}
        response = requests.get(
            url,
            headers=headers,
            params={
                "q": self.query,
                "extra_snippets": True
            }
        )
        return response.json()["web"]["results"][0:5]


providers = [WebSearchProvider]

with OpenAIChatModel(name="gpt-4o-mini") as llm:
    resp = llm.ask(
        prompt="Who won the 400m men individual medley at the 2024 Olympics?",
        tools=providers
    )

print(resp.choices[0].message.content)
```


#### Using tools in conjunction with structured output

Leveraging the structured output capabilities of PyLLMCore, you can combine both
the use of tools the generation of an object.

This means we can make the LLM use a tool to enrich its context and at the same time produce a structured output.

Here's a simple example on how you can add computational capabilities:

```python
import hashlib
from enum import Enum
from dataclasses import dataclass
from llm_core.assistants import OpenAIAssistant


HashFunction = Enum("HashFunction", ["sha512", "sha256", "md5"])


@dataclass
class HashProvider:
    hash_function: HashFunction
    content: str

    def __call__(self):
        hash_fn = getattr(hashlib, self.hash_function.name)
        return hash_fn(self.content.encode('utf-8')).hexdigest()


@dataclass
class Hash:
    system_prompt = "You are a helpful assistant"
    prompt = "{prompt}"

    hashed_content: str
    hash_algorithm: HashFunction
    hash_value: str

    @classmethod
    def ask(cls, prompt):
        with OpenAIAssistant(cls, tools=[HashProvider]) as assistant:
            response = assistant.process(prompt=prompt)
            return response

Hash.ask('Compute the sha256 for `py-llm-core`')
```


## Context window management

`py-llm-core` uses Tiktoken to estimate the length of strings in tokens. It is registered as a codec within the Python codecs registry :

```python
from llm_core.splitters import TokenSplitter
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
tokens = codecs.encode(text, 'tiktoken')
token_length = len(tokens)

# Chunking and splitting
splitter = TokenSplitter(
    chunk_size=50,
    chunk_overlap=0
)

for chunk in splitter.chunkify(text):
    print(chunk)
```

## Classification and using enums

One useful use case when interacting with LLMs is their ability to understand 
what a user wants to achieve using natural language.

Here's a simplified example :

```python
from enum import Enum
from dataclasses import dataclass
from llm_core.assistants import OpenWeightsAssistant

class TargetItem(Enum):
    PROJECT = 1
    TASK = 2
    COMMENT = 3
    MEETING = 4


class CRUDOperation(Enum):
    CREATE = 1
    READ = 2
    UPDATE = 3
    DELETE = 4


@dataclass
class UserQuery:
    system_prompt = "You are a helpful assistant."
    prompt = """
    Analyze the user's query and convert his intent to:
    - an operation (among CRUD)
    - a target item

    Query: {prompt}
    """
    operation: CRUDOperation
    target: TargetItem


def ask(prompt):
    with OpenWeightsAssistant(UserQuery, model="llama-8b-3.1-q4", loader_kwargs={"n_ctx": 4_000}) as assistant:
        user_query = assistant.process(prompt=prompt)
        return user_query

ask('Cancel all my meetings for the week')
```

## Note

PyLLMCore v3.0.0 comes with breaking changes. The goal behind the v3 is to be able to take
full advantages of LLMs as fast as possible (documentation takes 5 min to read).

For developers using the 2.x versions, you may want to stick to the latest 2.8.15 version. However the 2.x branch won't be maintained (open an issue if you need help migrating your project).

The latest version comes with major simplifications to be even easier to use.

See the following quick start guide.


## Changelog

- 3.4.4: Improved the tool use prompting and structure
- 3.4.3: Disabled parallel_tool_calls
- 3.4.2: Fixed bug when using more than one tool
- 3.4.1: Fixed bug when building field type name
- 3.4.0: Fixed prompt when using tools
- 3.3.0: Added support for Python 3.8
- 3.2.0: Added support for Anthropic models
- 3.1.0:
    - Added back support for Azure OpenAI
    - Unified the way to load language models (API or Open Weights)
- 3.0.0:
    - Simplified the code and the documentation
    - Upgraded Mistral AI dependencies (Use `MistralAIModel` class)
    - Simplified management of tokens
    - Dropped Azure AI support
    - Dropped LLaVACPPModel support
    - Dropped NuExtract support
    - Moved assistant implementations to a separate package
    - Refactored API gated model code
    - Renamed llama_cpp_compatible to open_weights
- 2.8.15: Fixed a bug when using only one tool
- 2.8.13: Rewrite of the function calling to add support for tools (OpenAI and LLaMA compatible)
- 2.8.11: Add support for NuExtract models
- 2.8.10: Add gpt-4o-2024-05-13
- 2.8.5: Fix model path building
- 2.8.4: Added support for Mistral Large
- 2.8.3: Raised timeout
- 2.8.1: Fixed bug when deserializing instances
- 2.8.0: Added support for native type annotation (pep585) for lists and sets
- 2.7.0: Fixed bug when function_call was set at None
- 2.6.1: Add dynamic max_tokens computation for OpenAI
- 2.6.0: Add support for Azure OpenAI
- 2.5.1: Fix bug on system prompt format
- 2.5.0: Add support for LLaVA models
- 2.4.0:
    + Set timeouts on OpenAI API
- 2.2.0:
    + Default settings on ARM64 MacOS modified (1 thread / offloading everything on the GPU)
    + Added `completion_kwargs` for Assistants to set temperature
- 2.1.0:
    + Added support for Enum to provide better support for classification tasks
    + Added example in the documentation
- 2.0.0:
    + Refactored code
    + Dynamically enable GPU offloading on MacOS
    + Added configuration option for storing local models (MODELS_CACHE_DIR)
    + Updated documentation

- 1.4.0: Free up resources in LLamaParser when exiting the context manager
- 1.3.0: Support for LLaMA based models (llama, llama2, Mistral Instruct)
- 1.2.0: Chain of density prompting implemented with OpenAI
- 1.1.0: Chain of Verification implemented with OpenAI
- 1.0.0: Initial version
