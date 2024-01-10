# PyLLMCore

## Overview

PyLLMCore is a light-weighted structured interface with Large Language Models 
with native support for [llama.cpp](http://github.com/ggerganov/llama.cpp) and OpenAI API.

The design decisions behind PyLLMCore are:

- Sane defaults
- Clear abstractions and terminology
- Out of the box utility classes

## Main benefits of using PyLLMCore

- Pythonic API
- Simple to use
- You need structures *everywhere* (provided by the standard library `dataclasses` module)
- High-level API with the `assistants` module
- Switching between models has never been easier

## Why you shouldn't use PyLLMCore

- You need a lot of external integrations: Take a look at [langchain](https://github.com/langchain-ai/langchain)
- You need tremendous performance: Take a look at [vllm](https://github.com/vllm-project/vllm)
- You don't need OpenAI: Take a look a [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) (which is integrated in PyLLMCore)
- You use Pydantic and don't use the dataclasses module


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

model = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

with LLaMACPPParser(Book, model=model) as parser:
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

#### Using Azure OpenAI

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


with OpenAIParser(Book, model="gpt-3.5-turbo-0613") as parser:
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

### Perform advanced tasks

#### Overview

To perform generic tasks, you will use the `assistants` module that provides generic assistants:

- `assistants.OpenAIAssistant`
- `assistants.LLaMACPPAssistant`

Using these assistants, you can take a look at how the utilities are built:

- `assistants.analysts.Analyst`
- `assistants.verifiers.Doubter`
- `assistants.verifiers.ConsistencyVerifier`
- `assistants.summarizers.Summarizer`


#### Create your own utility

There are 3 items required to build and run a utility:

- A language model (any compatible model)
- An assistant class: This is where your logic is written
- A results class: This is the structure you need. It also contains the prompt.

Here is an example where `Recipe` is the results class. We'll use the 
Mistral AI Instruct model.

```python
from typing import List
from dataclasses import dataclass

# LLaMACPPAssistant is needed to instanciate Mistral Instruct
from llm_core.assistants import LLaMACPPAssistant

# Make sure that ~/.cache/py-llm-core/models contains the following file
model = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"


@dataclass
class RecipeStep:
    step_title: str
    step_instructions: str

@dataclass
class Recipe:
    system_prompt = "You are a world-class chef"
    prompt = "Write a detailed step-by-step recipe to make {dish}"

    title: str
    steps: List[RecipeStep]
    ingredients: List[str]


class Chef:
    def generate_recipe(self, dish):
        with LLaMACPPAssistant(Recipe, model=model) as assistant:
            recipe = assistant.process(dish=dish)
            return recipe

chef = Chef()
recipe = chef.generate_recipe("Boeuf bourguignon")
print(recipe)

```

```python
Recipe(
    title="Boeuf Bourguignon Recipe",
    steps=[
        RecipeStep(
            step_title="Preheat the Oven",
            step_instructions="Preheat the oven to 350°F.",
        ),
        RecipeStep(
            step_title="Brown the Brisket",
            step_instructions="In a large pot, heat the olive oil over me...",
        ),
        RecipeStep(
            step_title="Cook the Onions and Garlic",
            step_instructions="Remove the brisket from the pot and set it...",
        ),
        RecipeStep(
            step_title="Simmer the Wine",
            step_instructions="Add the red wine to the pot and stir to sc...",
        ),
        RecipeStep(
            step_title="Bake in the Oven",
            step_instructions="Return the brisket to the pot, along with ...",
        ),
        RecipeStep(
            step_title="Finish Cooking",
            step_instructions="After 2 hours, remove the aluminum foil an...",
        ),
        RecipeStep(
            step_title="Serve",
            step_instructions="Remove the brisket from the pot and let it...",
        ),
    ],
    ingredients=[
        "1 pound beef brisket",
        "2 tablespoons olive oil",
        "1 large onion, chopped",
        "3 cloves garlic, minced",
        "1 cup red wine",
        "4 cups beef broth",
        "2 cups heavy cream",
        "1 teaspoon dried thyme",
        "1 teaspoon dried rosemary",
        "Salt and pepper to taste",
    ],
)


```

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

The partial summary generated is:

```python
SimpleSummary(
    content="""The Foundation series is a science fiction book series written
        by Isaac Asimov. It was first published as a series of short stories and
        novellas in 1942-50, and subsequently in three collections in 1951-53.
        ...
    """
)
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

model = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
assistant_cls = LLaMACPPAssistant

# Utilities
analyst = Analyst(model, assistant_cls)
doubter = Doubter(model, assistant_cls)
verifier = ConsistencyVerifier(model, assistant_cls)

# Fetch some content 
splitter = TokenSplitter(model=model, chunk_size=3_000)
pizza_dough_recipe = requests.get(pizza_dough_recipe_url).text
context = splitter.first_extract(pizza_dough_recipe)


query = "Write 3 advices when making pizza dough."

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

Here is a summary of what's been printed:

```txt
> Baseline answer:

When making pizza dough, it is important to choose high-protein flour such as bread or all-purpose flour.
The dough should be mixed and kneaded for a long time to develop flavor and gluten.
It is also important to let the dough rest and rise before shaping it into pizza balls.

> Questions

1. Is bread or all-purpose flour a good choice for making pizza dough?
2. How long should the dough be mixed and kneaded for flavor development and gluten formation?
3. Should the dough be allowed to rest and rise before shaping it into pizza balls?
4. What is the purpose of mixing and kneading the dough?
5. Is there a specific step in making pizza dough that can be skipped?

> Consistency checks

1.

Bread or all-purpose flour is a good choice for making pizza dough.
The rule of thumb is to pick a flour that has high protein content.

AnswerConsistency(is_consistent=True, is_inferred_from_context=True)


2.

The dough should be mixed and kneaded for around 5 minutes.
The mixing process starts the germination of the flour, which develops the flavor of the dough.
Kneading helps to form the gluten network that gives the dough its elasticity and structure.

AnswerConsistency(is_consistent=True, is_inferred_from_context=True)

...

```

From there, you can further process answers to remove any hallucinations or inconsistencies.


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

Tokenizers are registered as a codecs within the Python
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


# tokens = codecs.encode(text, 'gpt-3.5-turbo')
tokens = codecs.encode(text, 'mistral-7b-instruct-v0.1.Q4_K_M.gguf')

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

```

```python
In [2]: ask('Cancel all my meetings for the week')
Out[2]: UserQuery(operation=<CRUDOperation.DELETE: 4>, target=<TargetItem.MEETING: 4>)

In [3]: ask('What is the agenda ?')
Out[3]: UserQuery(operation=<CRUDOperation.READ: 2>, target=<TargetItem.MEETING: 4>)

In [4]: ask('Schedule meeting for next monday')
Out[4]: UserQuery(operation=<CRUDOperation.CREATE: 1>, target=<TargetItem.MEETING: 4>)

In [5]: ask('When is my next meeting ?')
Out[5]: UserQuery(operation=<CRUDOperation.READ: 2>, target=<TargetItem.MEETING: 4>)

# The classification went wrong here, so I tried a different formulation
In [6]: ask('Todo: read the final report on the project LLMCore')
Out[6]: UserQuery(operation=<CRUDOperation.READ: 2>, target=<TargetItem.TASK: 2>)

# Still no joy
In [7]: ask('Task: read the final report on the project LLMCore')
Out[7]: UserQuery(operation=<CRUDOperation.READ: 2>, target=<TargetItem.PROJECT: 1>)

# Being just a little more specific and voilà !
In [8]: ask('Add to my todo: read the final report on the project LLMCore')
Out[8]: UserQuery(operation=<CRUDOperation.CREATE: 1>, target=<TargetItem.TASK: 2>)
```


## Synthetic dataset generation example

```python
from typing import List
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
    user_queries: List[str]

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

```markdown
**Claim**: All forms of CMC should be studied in order to fully understand how online communication effects relationships

**Grounds**: Numerous studies have been conducted on various facets of Internet relationships, focusing on the levels of intimacy, closeness, different communication modalities, and the frequency of use of computer-mediated communication (CMC). However, contradictory results are suggested within this research mostly because only certain aspects of CMC are investigated.

**Warrant**: CMC is defined and used as ‘email’ in creating feelings of closeness or intimacy. Other articles define CMC differently and, therefore, offer different results.

**Qualifier**: The strength of the relationship was predicted best by FtF and phone communication, as participants rated email as an inferior means of maintaining personal relationships as compared to FtF and phone contacts.

**Rebuttal**: Other studies define CMC differently and, therefore, offer different results.

**Backing**: Cummings et al.'s (2002) research in relation to three other research articles
```

## LLaVA - Multi modalities - Mistral Vision

We can use a quantized version of the BakLLaVA model from [SkunkworksAI](https://github.com/SkunkworksAI) to process images.

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

```python
ChatCompletion(
    id='chatcmpl-c1e49a42-fe96-49ba-a47b-991479f7d672',
    object='chat.completion',
    created=1699476989,
    model='/Users/pas/.cache/py-llm-core/models/BakLLaVA-1-Q4_K_M.gguf',
    choices=[
        ChatCompletionChoice(
            index=0,
            message=Message(
                role='assistant',
                content='''The image features a brown background with large,
                  bold text that reads
                  "Understand, Learn, Build and Deploy LLM Projects."

                  This text is yellow.

                  The words "Leverage AI Power Without Disclosing Your Data"
                  are also written on the background, adding more information to the scene.
            '''),
            finish_reason='stop'
        )
    ],
    usage=Usage(
        prompt_tokens=630,
        completion_tokens=62,
        total_tokens=692
    ),
    system_fingerprint=None
)
```
