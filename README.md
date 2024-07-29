# PyLLMCore

## Overview

PyLLMCore is a light-weighted interface with Large Language Models.

It comes with native support:
- OpenAI (Official API + Azure)
- MistralAI (Official API + Azure)
- Open weights models (GGUF) thanks to `llama-cpp-python` bindings

## Expected benefits and reasons to use PyLLMCore

- Simple to use and to understand
- Pythonic API
- As little dependencies as possible
- Structures are *everywhere* provided by the standard library `dataclasses` module
- Easy swapping between models
- High-level API with the `assistants` module (these higher-level utility classes may be moved to a dedicated package)

## Why you shouldn't use PyLLMCore

- You need a whole framework: Take a look at [langchain](https://github.com/langchain-ai/langchain)
- You need tremendous performance: Take a look at [vllm](https://github.com/vllm-project/vllm)
- You want/need to use Pydantic and don't use the `dataclasses` module
- You need Python 3.8 or older (PyLLMCore requires at least 3.9)

## Use cases

PyLLMCore has evolved to covers a wider range of use cases and serves as a building brick:

- Parsing: see the `parsers` module
- Function calling or using tools
- Context size management: see the `splitters` module
- Tokenizing: see the `token_codecs` module
- Summarizing: see the `assistants.summarizers` module
- Question answering: see the `assistants.analyst` module
- Hallucinations reduction: see the `assistants.verifiers` module


## Install

### Quick start

```shell
pip install py-llm-core

#: To use OpenAI models, set your API key
export OPENAI_API_KEY=sk-<replace with your actual api key>

#: To use local models (i.e. offline - privately),
#: store your models in ~/.cache/py-llm-core/models

#: The following downloads the best models (you can use any GGUF models)
#: LLaMA-3.1-8B (Quantized version Q4_K_M)
#: NuExtract  (Quantized version Q8)
#: Mistral 7B v0.3 (Quantized version Q4_K_M)

mkdir -p ~/.cache/py-llm-core/models
wget -O ~/.cache/py-llm-core/models/llama-8b-3.1-q4 \
    https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf?download=true

wget -O ~/.cache/py-llm-core/models/nuextract-q8 \
    https://huggingface.co/advanced-stack/NuExtract-GGUF/resolve/main/nuextract-q8.gguf?download=true

wget -O ~/.cache/py-llm-core/models/mistral-7b-v0.3-q4 \
    https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf?download=true
```

## Documentation

### Parsing

You can use these following examples to extract information from raw text.

```python
from dataclasses import dataclass

from llm_core.parsers import LLaMACPPParser

# : You can use any one of the other available parsers
# from llm_core.parsers import NuExtractParser (Smaller models available)
# from llm_core.parsers import OpenAIParser (Requires an API key)


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

with LLaMACPPParser(Book, model="mistral-7b-v0.3-q4") as parser:
    book = parser.parse(text)
    print(book)

#: For NuExtractParser, the fields of the Book class shall have default values
# with NuExtractParser(Book, model="nuextract-q8") as parser:
#     book = parser.parse(text)
#     print(book)

# with OpenAIParser(Book, model="gpt-4o-mini") as parser:
#     book = parser.parse(text)
#     print(book)
```


### Perform advanced tasks

#### Using tools a.k.a. Function calling

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

        # Returns the top 5 results
        return response.json()["web"]["results"][0:5]


providers = [WebSearchProvider]

llm = OpenAIChatModel(name="gpt-4o-mini")
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


HashFunction = Enum("HashFunction", ["sha256", "md5"])


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
        with OpenAIAssistant(cls, model="gpt-3.5-turbo") as assistant:
            assistant.tools = [HashProvider]
            response = assistant.process(prompt=prompt)
            return response

Hash.ask('Compute the sha for `py-llm-core`')

Hash(
    hashed_content='py-llm-core',
    hash_algorithm=<HashFunction.sha256: 1>,
    hash_value='38ec92d973268cb671e9cd98a2f5a7b8c4d451d87b61670c1fe236fd7777f708'
)
```

#### Overview

The `assistants` module provides a concise syntax to create features using both
large language models capabilities and structured outputs.

You can see examples in the source code:

- `assistants.analysts.Analyst`
- `assistants.verifiers.Doubter`
- `assistants.verifiers.ConsistencyVerifier`
- `assistants.summarizers.Summarizer`


#### Create your assistant class

In this example, we create an assistant that takes a dish as an input and
generate recipes in a structured manner.

```python
from dataclasses import dataclass
from llm_core.assistants import LLaMACPPAssistant


@dataclass
class RecipeStep:
    step_title: str
    step_instructions: str


@dataclass
class Recipe:
    system_prompt = "You are a world-class chef"
    prompt = "Write a detailed step-by-step recipe to make {dish}"

    title: str
    steps: list[RecipeStep]
    ingredients: list[str]

    @classmethod
    def from_dish(cls, dish):
        with LLaMACPPAssistant(cls, model="mistral-7b-v0.3-q4") as assistant:
            recipe = assistant.process(dish=dish)
            return recipe


recipe = Recipe.from_dish("Boeuf bourguignon")
print(recipe)
```

#### Summarizing

```python
import wikipedia
from llm_core.assistants import Summarizer, LLaMACPPAssistant


summarizer = Summarizer(
    model="mistral-7b-v0.3-q4",
    assistant_cls=LLaMACPPAssistant
)

text = wikipedia.page("Foundation from Isaac Asimov").content

# To summarize only with 50% of the model context size
partial_summary = summarizer.fast_summarize(text)

# Iterative summaries on the whole content
for summary in summarizer.summarize(text):
    print(summary)
```


#### Reduce hallucinations using the verifiers module

This example implements loosely the Chain of Verification (CoVe).

To reduce hallucinations in the LLM completions, you can use the following example
as a starting point:

```python
import requests
from llm_core.splitters import TokenSplitter
from llm_core.assistants import (
    Analyst,
    Doubter,
    ConsistencyVerifier,
    LLaMACPPAssistant,
)

pizza_dough_recipe_url = (
    "https://raw.githubusercontent.com/hendricius/pizza-dough/main/README.md"
)

model = "mistral-7b-v0.3-q4"
assistant_cls = LLaMACPPAssistant

# Utilities
analyst = Analyst(model, assistant_cls)
doubter = Doubter(model, assistant_cls)
verifier = ConsistencyVerifier(model, assistant_cls)

# Fetch some content 
splitter = TokenSplitter(model=model, chunk_size=4_000)
pizza_dough_recipe = requests.get(pizza_dough_recipe_url).text
context = splitter.first_extract(pizza_dough_recipe)


query = "Write 3 recommendations on how to make the best pizza dough."

analyst_response = analyst.ask(query, context)
question_collection = doubter.verify(query, analyst_response.content)
questions = question_collection.questions

answers = []

for question in questions:
    response = analyst.ask(question, context=context)
    answers.append(response.content)

for question, answer in zip(questions, answers):
    verifications = verifier.verify(
        question=question, context=context, answer=response.content
    )
```


## Tokenizer

Tokenizers are registered as a codecs within the Python codecs registry :

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
# tokens = codecs.encode(text, 'gpt-3.5-turbo')
tokens = codecs.encode(text, 'mistral-7b-v0.3-q4')

token_length = len(tokens)
print(token_length)

# Chunking and splitting


splitter = TokenSplitter(
    model="mistral-7b-v0.3-q4",
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
from dataclasses import dataclass
from llm_core.assistants import LLaMACPPAssistant
from enum import Enum

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
    with LLaMACPPAssistant(UserQuery, model="mistral") as assistant:
        user_query = assistant.process(prompt=prompt)
        return user_query

ask('Cancel all my meetings for the week')
```


## Synthetic dataset generation example

```python
from typing import list
from enum import Enum
from dataclasses import dataclass
from llm_core.assistants import LLaMACPPAssistant


class Item(Enum):
    CALENDAR = 1
    EVENT = 2
    TASK = 3
    REMINDER = 4
    INVITEE = 5


class CRUDOperation(Enum):
    CREATE = 1
    READ = 2
    UPDATE = 3
    DELETE = 4


@dataclass
class UserQueryGenerator:
    system_prompt = "You are a helpful assistant."
    prompt = """
    # Goals

    We are developing a new business calendar software that is able to understand plain english.
    
    # Examples
    
    Cancel all my meetings of the week
    What is my next meeting ?
    What is on the agenda for the meeting at 1 pm ?
    {queries}
    
    # Todo

    Write {queries_count} new examples of what a user could have asked.
    
    """
    user_queries: list[str]

    @classmethod
    def generate(cls, queries_count=10, existing_queries=()):
        with LLaMACPPAssistant(cls, model="mistral") as assistant:
            existing_queries_str = '\n'.join(existing_queries)
            batch = assistant.process(queries_count=queries_count, queries=existing_queries_str)
            return batch.user_queries


@dataclass
class UserQueryClassification:
    system_prompt = "You are a helpful assistant."
    prompt = """
    Analyze the user's query and convert his intent to:
    - an operation (among CRUD)
    - a target item

    Query: {prompt}
    """
    operation: CRUDOperation
    item: Item

    @classmethod
    def ask(cls, prompt):
        with LLaMACPPAssistant(cls, model="mistral") as assistant:
            user_query = assistant.process(prompt=prompt)
            return user_query
```


## Argument analysis using Toulmin's method

See the code in `examples/toulmin-model-argument-analysis.py`

```shell
python3 examples/toulmin-model-argument-analysis.py
```

## LLaVA - Multi modalities - Mistral Vision

SkunkworksAI released the BakLLaVA model, which is a Mistral 7B instruct model augmented with vision capabilities [SkunkworksAI](https://github.com/SkunkworksAI).

Download `BakLLaVA-1-Q4_K_M.gguf` and `BakLLaVA-1-clip-model.gguf` files from https://huggingface.co/advanced-stack/bakllava-mistral-v1-gguf/tree/main

To run inference:

```python
from llm_core.llm import LLaVACPPModel

model = "BakLLaVA-1-Q4_K_M.gguf"

llm = LLaVACPPModel(
    name=model,
    llama_cpp_kwargs={
        "logits_all": True,
        "n_ctx": 8000,
        "verbose": False,
        "n_gpu_layers": 100, # Set to 0 if you don't have a GPU
        "n_threads": 1, 
        "clip_model_path": "BakLLaVA-1-clip-model.gguf"
    }
)

llm.load_model()

history = [
    {
        'role': 'user',
        'content': [
            {'type': 'image_url', 'image_url': 'http://localhost:8000/adv.png'}
        ]
    }
]

llm.ask('Describe the image as accurately as possible', history=history)

```

## Using Azure OpenAI

Using OpenAI Azure services can be enabled by following the steps:

1. Create an Azure account
2. Enable Azure OpenAI cognitive services
3. Get the API key and the API endpoint provided
4. Set the environment variable USE_AZURE_OPENAI to True (`export USE_AZURE_OPENAI=True`)
5. Set the environment variable AZURE_OPENAI_ENDPOINT (see step 3)
6. Set the environment variable AZURE_OPENAI_API_KEY (see step 3)
7. Create a deployment where you use the model name from OpenAI. You'll need to remove dot signs, i.e. for the model `gpt-3.5-turbo-0613` create a deployment named `gpt-35-turbo-0613`.
8. PyLLMCore will take care of removing the dot sign for you so you can use the same code base for both OpenAI and Azure.
9. When calling Parser or Assistant classes, specify the model

The following example uses Azure OpenAI:

```shell
export USE_AZURE_OPENAI=True
export AZURE_OPENAI_API_KEY=< your api key >
export AZURE_OPENAI_ENDPOINT=https://< your endpoint >.openai.azure.com/
```

## Troubleshooting

### llama-cpp-installation failure (legacy)

The following workaround should no longer be necessary:

The `llama-cpp-python` dependency may improperly detects the architecture and raise an error `an incompatible architecture (have 'x86_64', need 'arm64'))`.

If that's the case, run the following in your virtual env:

```shell
CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64" pip3 install --upgrade --verbose --force-reinstall --no-cache-dir llama-cpp-python
```


## Changelog

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
