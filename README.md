# PyLLMCore

## Overview

PyLLMCore is a lightweight Python library designed to provide a simple and efficient interface for interacting with Large Language Models (LLMs). It supports a variety of models, including:

- **OpenAI**: Access state-of-the-art models like GPT-4o.
- **MistralAI**: Use models optimized for specific tasks.
- **Anthropic**: Engage with Claude.
- **Google AI (Gemini)**: Leverage the largest context window of Google's Gemini serie.
- **Open-Weights Models (GGUF)**: Use a wide range of open-source models via `llama-cpp-python` bindings.


### Key Features

- **Pythonic API**: Designed to be intuitive and easy to use for Python developers.
- **Minimal dependencies**: Built with as few dependencies as possible to ensure ease of installation and integration.
- **Flexible model switching**: Easily swap between different models to suit your needs.
- **Standard library integration**: Uses Python's `dataclasses` for structured data handling.

### System Requirements

- **Python 3.8** or higher is required to use PyLLMCore.


### Why Choose PyLLMCore?

- **Ease of use**: Simple setup and usage make it accessible for developers of all levels.
- **Versatility**: Supports a wide range of models and use cases, from parsing text to function calling.
- **Customization**: Offers the ability to extend and customize functionality with minimal effort.

### When to consider alternatives

- If you need a comprehensive framework, consider [LangChain](https://github.com/langchain-ai/langchain).
- For high-performance requirements, explore [vllm](https://github.com/vllm-project/vllm).
- If you prefer using Pydantic over `dataclasses`, PyLLMCore might not be the best fit.


## Use Cases

PyLLMCore is versatile and can be used in various scenarios involving Large Language Models (LLMs). Here are some common use cases:

1. **Parsing raw content:**
   - Use the `parsers` module to extract structured information from unstructured text. This is useful for applications like data extraction, content analysis, and information retrieval.

2. **Tool and function calling:**
   - Leverage the `assistants` module to enable LLMs to interact with external tools and functions. This can enhance the model's capabilities by integrating with APIs or performing specific tasks.

3. **Context window size management:**
   - Utilize the `splitters` module to manage large text inputs by splitting them into manageable chunks. This is particularly useful when dealing with small models that have context window limitations.

4. **Custom model integration:**
   - Easily switch between different LLMs, including OpenAI, MistralAI, Anthropic, Google AI, and open-weights models, to suit specific requirements or preferences.

5. **Advanced tasks:**
   - Implement advanced functionalities such as structured output generation, classification tasks, and more by customizing the library's features.


## Install

### Prerequisites

- Ensure you have **Python 3.8** or higher installed on your system.
- It's recommended to use a virtual environment to manage dependencies.

### Installation steps

1. **Set up a virtual environment (optional but recommended):**

   ```shell
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install PyLLMCore:**

   Use `pip` to install the library:

   ```shell
   pip install py-llm-core
   ```

3. **Configure API keys:**

   If you plan to use OpenAI models, set your API key as an environment variable:

   ```shell
   export OPENAI_API_KEY=sk-<replace with your actual api key>
   ```

   For Azure OpenAI, set the following environment variables:

   ```shell
   export AZURE_OPENAI_API_KEY=<your-azure-api-key>
   export AZURE_OPENAI_ENDPOINT=<your-azure-endpoint>
   export AZURE_OPENAI_API_VERSION=<api-version>
   ```

   For MistralAI set the respective API keys:

   ```shell
   export MISTRAL_API_KEY=<your-mistral-api-key>
   ```

   For Anthropic set the respective API keys:

   ```shell
   export ANTHROPIC_API_KEY=<your-anthropic-api-key>
   ```

   For Google AI set the respective API keys:

   ```shell
   export GOOGLE_API_KEY=<your-google-api-key>
   ```

4. **Download local models (Optional):**

   If you want to use local open-weights models offline, download and store them in the specified directory:

   ```shell
   mkdir -p ~/.cache/py-llm-core/models
   wget -O ~/.cache/py-llm-core/models/llama-8b-3.1-q4 \
       https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf?download=true

   wget -O ~/.cache/py-llm-core/models/mistral-7b-v0.3-q4 \
       https://huggingface.co/lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf?download=true
   ```

### Quick Start

Explore the Jupyter notebook in the `/notebooks/` directory for executable examples to get started quickly.

## Documentation

### Parsing with PyLLMCore

The `py-llm-core` library provides a straightforward way to parse and extract structured information from unstructured text using various Large Language Models (LLMs). Below are examples of how to use the `OpenAIParser` and how to switch between different parsers.

#### Basic Example with OpenAIParser

To parse text using OpenAI models, you can use the `OpenAIParser`. Here's a simple example:

```python
from dataclasses import dataclass
from typing import List
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

text = """The Foundation series is a science fiction book series written by
American author Isaac Asimov. First published as a series of short
stories and novellas in 1942–50, and subsequently in three books in
1951–53, for nearly thirty years the series was widely known as The
Foundation Trilogy: Foundation (1951), Foundation and Empire (1952),
and Second Foundation (1953)."""

with OpenAIParser(BookCollection) as parser:
    books_collection = parser.parse(text)

    for book in books_collection.books:
        print(book)
```

Now, you can parse images when using a compatible model (tested with OpenAI `gpt-4o-mini` and Mistral `pixtral-12b-2409`)

```python
from dataclasses import dataclass
from llm_core.parsers import OpenAIParser
from llm_core.parsers import MistralAIParser
from llm_core.loaders import load_image


@dataclass
class Receipt:
    title: str
    expense_category: str
    tax_amount: float
    total_amount_without_tax: float
    total_amount_with_tax: float


image_b64 = load_image(
    "/path/to/a/receipt.jpg"
)

with OpenAIParser(Receipt) as parser:
    receipt = parser.parse(image_b64=image_b64)

print(receipt)

#: With Mistral Pixtral

with MistralAIParser(Receipt, model="pixtral-12b-2409") as parser:
    receipt = parser.parse(image_b64=image_b64)

print(receipt)
```

#### Advanced Example with OpenAIParser

For more complex parsing tasks, you can define a more detailed schema:

```python
from dataclasses import dataclass
from typing import List, Dict
from llm_core.parsers import OpenAIParser

@dataclass
class Book:
    title: str
    summary: str
    author: str
    published_year: int
    awards: List[str]
    genres: List[str]

@dataclass
class BookCollection:
    books: List[Book]

text = """The Foundation series by Isaac Asimov includes several award-winning books
such as Foundation (1951), which won the Hugo Award. The series spans genres like
science fiction and speculative fiction."""

with OpenAIParser(BookCollection) as parser:
    books_collection = parser.parse(text)

    for book in books_collection.books:
        print(book)
```

#### Switching Parsers

You can easily switch between different parsers to use models from other providers:

- **MistralAIParser**: For MistralAI models.
```python
from llm_core.parsers import MistralAIParser
with MistralAIParser(BookCollection) as parser:
  books_collection = parser.parse(text)
```

- **OpenWeightsParser**: For open-weights models.
```python
from llm_core.parsers import OpenWeightsParser
with OpenWeightsParser(BookCollection) as parser:
  books_collection = parser.parse(text)
```

- **AnthropicParser**: For Anthropic models.
```python
from llm_core.parsers import AnthropicParser
with AnthropicParser(BookCollection) as parser:
  books_collection = parser.parse(text)
```

- **GoogleAIParser**: For Google AI models.
```python
from llm_core.parsers import GoogleAIParser
with GoogleAIParser(BookCollection) as parser:
  books_collection = parser.parse(text)
```

### Working with Open Weights Models

PyLLMCore allows you to work with open weights models, providing flexibility to use models offline. To use these models, follow these steps:

1. **Model Location**: By default, models are stored in the `~/.cache/py-llm-core/models` directory. You can change this location by setting the `MODELS_CACHE_DIR` environment variable.

2. **Model Selection**: To select an open weights model, specify the model name when initializing the `OpenWeightsModel` class. Ensure the model file is present in the specified directory. For example:

```python
from llm_core.llm import OpenWeightsModel

model_name = "llama-8b-3.1-q4"  # Replace with your model's name
with OpenWeightsModel(name=model_name) as model:
   # Use the model for your tasks
   pass
```

Ensure that the model file, such as `llama-8b-3.1-q4`, is downloaded and stored in the `MODELS_CACHE_DIR`.

3. **Downloading Models**: You can download models from sources like Hugging Face. Use the following command to download a model:

```shell
   wget -O ~/.cache/py-llm-core/models/llama-8b-3.1-q4 \
       https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf?download=true
```



### Advanced Tasks

The `py-llm-core` library offers advanced capabilities to enhance the functionality of Large Language Models (LLMs). Below are some key features and examples to help you leverage these capabilities effectively.

#### Using Tools (Function Calling)

Enhance LLM responses by integrating external tools. Define a tool using a dataclass with a `__call__` method to implement the desired logic.


Here's an example of retrieving one's public IP address:

```python
import requests
from decouple import config
from dataclasses import dataclass
from llm_core.llm import OpenAIChatModel

@dataclass
class PublicIPProvider:

    def __call__(self):
        url = "https://ipv4.jsonip.com"
        response = requests.get(url).json()['ip']
        return response

providers = [PublicIPProvider]

with OpenAIChatModel(name="gpt-4o-mini") as llm:
    response = llm.ask(prompt="What's my IP ?", tools=providers)

print(response.choices[0].message.content)
```

Here's an example of adding web search capabilities using the Brave Search API:

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
        response = requests.get(url, headers=headers, params={"q": self.query})
        return response.json()["web"]["results"][0:5]

providers = [WebSearchProvider]

with OpenAIChatModel(name="gpt-4o-mini") as llm:
    response = llm.ask(prompt="Who won the 400m men individual medley at the 2024 Olympics?", tools=providers)

print(response.choices[0].message.content)
```

#### Combining Tools with Structured Output

You can combine tool usage with structured output generation. This allows the LLM to use a tool and produce a structured response simultaneously. Here's an example of adding computational capabilities:

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

Hash.ask('Compute the sha256 for the string `py-llm-core`')
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


## Changelog

- 3.5.0: Added Vision support
- 3.4.13: Disabled parallel_tool_calls (improved)
- 3.4.12: Fixed export of AzureOpenAIAssistant
- 3.4.11: Updated loader_kwargs override
- 3.4.10: Added helpers for Azure OpenAI models
- 3.4.9: Added suppport for Google AI Gemini
- 3.4.8: Removed unsupported attributes for Usage
- 3.4.7: Added support for `completion_tokens_details`
- 3.4.6: Fixed a bug appearing when the LLM does not want to use any tool
- 3.4.5:
    - Fixed parallel_tool_calls bug
    - Added support for `raw_tool_results` argument in `ask` to stop generation
      and output unprocessed tool results.
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
