# API Reference: `llm_core.assistants`

Assistants in PyLLMCore are high-level components for managing complex interactions with LLMs, often involving tool usage (function calling) and sophisticated prompting strategies. They build upon the functionality of Models and Parsers.

## `llm_core.assistants.base.BaseAssistant`

This is the base class for all assistants. It orchestrates the interaction flow, including prompt formatting, tool integration, and processing the LLM's response into a structured output.

**Inheritance:**

*   `BaseAssistant` inherits from `llm_core.parsers.BaseParser`. This means an Assistant *is a* Parser and utilizes its capabilities for schema definition (`target_cls`) and deserialization. It also means it inherits constructor parameters like `model`, `model_cls`, `loader`, and `loader_kwargs`.

**Key Constructor Parameters (in addition to those from `BaseParser`):**

*   `target_cls (Callable)`: **Required.** A Python dataclass that serves two purposes:
    1.  Defines the schema for the final structured output of the assistant (similar to how Parsers use it).
    2.  Can optionally provide `system_prompt: str` and `prompt: str` as class attributes. The `BaseAssistant` will use these as templates for interacting with the LLM.
*   `tools (list)`: Optional. A list of tool definitions that the assistant can provide to the LLM. Each tool is typically a dataclass with a `__call__(self)` method that implements the tool's logic. Default: `None`.
*   `system_prompt (str)`: Overrides the `system_prompt` from `target_cls` or `BaseParser` if provided directly. Default: `"You are a helpful assistant"`.
*   `prompt (str)`: Overrides the `prompt` from `target_cls` if provided directly. Default: `""`.

**`__post_init__` Behavior:**

*   Calls `super().__post_init__()` (which initializes the `BaseParser`, including setting up the underlying `llm` instance and `target_json_schema`).
*   It then attempts to retrieve `system_prompt` and `prompt` attributes from the `self.target_cls`. If found, these will be used as templates.

**Methods:**

*   `process(**kwargs) -> object`:
    *   The primary method to run the assistant's logic.
    *   **Parameters:**
        *   `**kwargs`: Keyword arguments that will be used to format the `system_prompt` and `user_prompt` templates (if they contain placeholders like `{placeholder_name}`).
    *   **Returns:** An instance of the `target_cls` populated with the LLM's response, or potentially a more direct tool output if the schema and tool interaction dictates.
    *   **Internal Logic:**
        1.  Formats `self.system_prompt` and `self.prompt` (obtained from `target_cls` or constructor) using the provided `**kwargs`.
        2.  Sets the formatted `system_prompt` on the underlying LLM instance (`self.llm.system_prompt`).
        3.  Calls `self.llm.ask()` with the formatted `prompt`, the `schema` derived from `target_cls`, and the provided `tools`.
            *   If `self.target_json_schema["properties"]` is empty (meaning the `target_cls` has no fields, e.g., an empty dataclass used just for its prompt attributes and potentially tool use without structured output), it calls `ask` with `raw_tool_results=True`. This is useful if the assistant's purpose is primarily to invoke a tool and the specific structured output isn't the main goal, or the tool itself returns all necessary information.
        4.  Deserializes the LLM's response (which is expected to be JSON conforming to `target_cls`'s schema) into an instance of `target_cls` using `self.deserialize()` (inherited from `BaseParser`).

**Tool Definition and Usage:**

*   Tools are provided as a list of dataclasses.
*   Each tool dataclass should have a `__call__(self)` method that contains the logic to execute the tool.
*   The fields of the tool dataclass define the parameters that the LLM can populate.
*   The underlying `LLMBase.ask()` method handles the complex parts of presenting tools to the LLM, interpreting the LLM's request to use a tool, populating its arguments, and calling the `__call__` method.

---

## Provider-Specific Assistant Classes

These classes inherit from `BaseAssistant` AND the corresponding provider-specific `Parser` class (e.g., `OpenAIAssistant` inherits from `BaseAssistant` and `OpenAIParser`). This multiple inheritance provides the assistant with both the general assistant logic and the specific model/parser configurations for a provider.

They primarily pre-configure the `model_cls` and a default `model` name. You typically need to provide the `target_cls` (and `tools` if applicable) when instantiating them.

*   **`llm_core.assistants.OpenAIAssistant(target_cls: Callable, model: str = "gpt-4o-mini", tools: list = None, ...)`**
    *   Uses `OpenAIChatModel`. Default model: `"gpt-4o-mini"`.
*   **`llm_core.assistants.AzureOpenAIAssistant(target_cls: Callable, model: str = "gpt-4o-mini", tools: list = None, ...)`**
    *   Uses `AzureOpenAIChatModel`. Default model: `"gpt-4o-mini"`.
*   **`llm_core.assistants.MistralAIAssistant(target_cls: Callable, model: str = "open-mistral-nemo", tools: list = None, ...)`**
    *   Uses `MistralAIModel`. Default model: `"open-mistral-nemo"`.
*   **`llm_core.assistants.OpenWeightsAssistant(target_cls: Callable, model: str = "mistral-7b-v0.3-q4", tools: list = None, ...)`**
    *   Uses `OpenWeightsModel`. Default model: `"mistral-7b-v0.3-q4"`.
*   **`llm_core.assistants.AnthropicAssistant(target_cls: Callable, model: str = "claude-3-5-sonnet-20240620", tools: list = None, ...)`**
    *   Uses `AnthropicModel`. Default model: `"claude-3-5-sonnet-20240620"`.
*   **`llm_core.assistants.GoogleAIAssistant(target_cls: Callable, model: str = "gemini-1.5-flash", tools: list = None, ...)`**
    *   Uses `GoogleAIModel`. Default model: `"gemini-1.5-flash"`.

**Example Usage (Conceptual - combining structured output and tools):**

```python
from dataclasses import dataclass, field
from enum import Enum
from llm_core.assistants import OpenAIAssistant # Or any other provider

# Tool Definition
class Unit(Enum):
    CELSIUS = "celsius"
    FAHRENHEIT = "fahrenheit"

@dataclass
class WeatherTool:
    location: str
    unit: Unit = Unit.CELSIUS

    def __call__(self):
        # In a real scenario, this would call a weather API
        if self.location == "Paris":
            return f"The weather in Paris is 20 degrees {self.unit.value}."
        return f"Weather data for {self.location} not found."

# Assistant's Target Dataclass (for output and prompting)
@dataclass
class WeatherReport:
    # Prompts are class attributes without type hints
    system_prompt = "You are a helpful weather assistant. Use tools to find weather information."
    prompt = "{prompt}"

    # Fields for structured output
    city: str
    temperature: str # e.g., "20 degrees celsius"
    summary: str

# Using the Assistant
available_tools = [WeatherTool]

with OpenAIAssistant(WeatherReport, tools=available_tools) as assistant:
    report = assistant.process(prompt="What's the weather like in Paris?")
    print(f"City: {report.city}")
    print(f"Temperature: {report.temperature}")
    print(f"Summary: {report.summary}")
```

**Note on Defining Prompts:**

When using a dataclass as `target_cls` for an assistant, the `system_prompt` and `prompt` attributes that provide templates should be defined as simple string class attributes directly within the dataclass, **without type hints**.

For example:
```python
@dataclass
class MyAssistantOutput:
    # Prompts are class attributes without type hints
    system_prompt = "This is the system message."
    prompt = "Process this: {user_input}"
    
    # Other fields for structured output (these should have type hints)
    processed_data: str 
```

Using `dataclasses.field` or including type hints (e.g., `system_prompt: str = "..."`) for these specific prompt attributes in the `target_cls` is incorrect for the purpose of providing templates to the `BaseAssistant`. These attributes are read as class-level variables from the `target_cls`, and the current implementation expects them without type annotations.

# Expected output might be (LLM dependent):
# City: Paris
# Temperature: 20 degrees celsius
# Summary: The weather in Paris is 20 degrees celsius.
```
This example illustrates how `target_cls` (`WeatherReport`) defines prompts and output structure, and how a `WeatherTool` can be used by the assistant.
