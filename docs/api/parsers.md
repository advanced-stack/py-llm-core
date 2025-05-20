# API Reference: `llm_core.parsers`

Parsers in PyLLMCore are designed to extract structured information (defined by a Python dataclass) from unstructured text or images. They leverage the underlying Model classes to perform this task.

## `llm_core.parsers.BaseParser`

This is the base class for all parsers. It handles the common logic of schema generation, prompting the LLM, and deserializing the output into a target dataclass instance.

**Key Constructor Parameters:**

*   `target_cls (Callable)`: **Required.** A Python dataclass that defines the structure of the information to be extracted. The parser will return an instance of this class.
*   `model (str)`: **Required.** The specific model identifier to be used by the underlying LLM for parsing (e.g., "gpt-4o-mini"). The availability of models depends on the `model_cls`.
*   `model_cls (Callable)`: **Required.** The LLM model class to be used (e.g., `OpenAIChatModel`, `MistralAIModel`). This determines which LLM provider will be used.
*   `loader (Callable)`: Optional. Passed to the `model_cls` constructor. Relevant for models requiring explicit loading (e.g., `OpenWeightsModel`). Default: `None`.
*   `loader_kwargs (dict)`: Optional. Passed to the `model_cls` constructor. Keyword arguments for the `loader`. Default: `None`.
*   `system_prompt (str)`: The system prompt to guide the LLM's parsing behavior. Default: `"Parse and process information from unstructured content."`.

**`__post_init__` Behavior:**

*   Converts the `target_cls` into a JSON schema using `llm_core.schema.to_json_schema`. This schema is used to instruct the LLM on the desired output format.
*   Initializes an instance of the specified `model_cls` with the given `model` name, `system_prompt`, and any `loader`/`loader_kwargs`.

**Context Manager Support:**

*   `__enter__(self)`: Loads the underlying model by calling `self.llm.load_model()`. Returns the parser instance. This makes parsers usable with Python's `with` statement.
*   `__exit__(self, exc_type, exc_val, exc_tb)`: Releases the underlying model by calling `self.llm.release_model()`.

**Methods:**

*   `parse(text: str = None, image_b64: str = None) -> object`:
    *   The core method for performing the parsing operation.
    *   **Parameters:**
        *   `text (str)`: Optional. The unstructured text content to parse.
        *   `image_b64 (str)`: Optional. A base64 encoded string of an image to parse (for multimodal models).
        *   At least one of `text` or `image_b64` should be provided.
    *   **Returns:** An instance of the `target_cls` populated with the extracted data.
    *   **Internal Logic:**
        1.  Constructs a detailed prompt for the LLM, including the input `text`/`image_b64` and the JSON schema of the `target_cls`.
        2.  Calls the `ask()` method of the underlying LLM instance (`self.llm`) with this prompt and schema.
        3.  The LLM's response (expected to be JSON) is then deserialized into an instance of `target_cls` using `llm_core.schema.from_dict` and `dirtyjson.loads` (which is robust to common LLM JSON formatting quirks).
*   `deserialize(json_str: str) -> object`:
    *   Takes a JSON string and attempts to convert it into an instance of `self.target_cls` using `dirtyjson.loads` and `from_dict`.
    *   Raises an exception if deserialization fails.

---

## Provider-Specific Parser Classes

These classes inherit from `BaseParser` and pre-configure it for a specific LLM provider by setting the `model_cls` and a default `model` name. You typically only need to provide the `target_cls` when instantiating them, unless you want to use a non-default model for that provider.

*   **`llm_core.parsers.OpenAIParser(target_cls: Callable, model: str = "gpt-4o-mini", ...)`**
    *   Uses `OpenAIChatModel`. Default model: `"gpt-4o-mini"`.
*   **`llm_core.parsers.AzureOpenAIParser(target_cls: Callable, model: str = "gpt-4o-mini", ...)`**
    *   Uses `AzureOpenAIChatModel`. Default model: `"gpt-4o-mini"`.
*   **`llm_core.parsers.MistralAIParser(target_cls: Callable, model: str = "open-mistral-nemo", ...)`**
    *   Uses `MistralAIModel`. Default model: `"open-mistral-nemo"`.
*   **`llm_core.parsers.OpenWeightsParser(target_cls: Callable, model: str = "mistral-7b-v0.3-q4", ...)`**
    *   Uses `OpenWeightsModel`. Default model: `"mistral-7b-v0.3-q4"`.
*   **`llm_core.parsers.AnthropicParser(target_cls: Callable, model: str = "claude-3-5-sonnet-20240620", ...)`**
    *   Uses `AnthropicModel`. Default model: `"claude-3-5-sonnet-20240620"`.
*   **`llm_core.parsers.GoogleAIParser(target_cls: Callable, model: str = "gemini-1.5-flash", ...)`**
    *   Uses `GoogleAIModel`. Default model: `"gemini-1.5-flash"`.

**Example Usage (Conceptual):**

```python
from dataclasses import dataclass
from llm_core.parsers import OpenAIParser # Or any other provider-specific parser

@dataclass
class UserInfo:
    name: str
    age: int

raw_text = "John Doe is 30 years old."

with OpenAIParser(UserInfo) as parser:
    user_info_instance = parser.parse(text=raw_text)
    print(user_info_instance.name) # Output: John Doe
    print(user_info_instance.age)  # Output: 30
```
This example demonstrates how to define a `target_cls` (`UserInfo`) and use a parser to extract and structure data from a simple text string.
