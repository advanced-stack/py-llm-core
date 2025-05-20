# API Reference: `llm_core.llm` (Models)

This section provides a detailed API reference for the model classes in `llm_core.llm`. Models are the core components responsible for direct interaction with Large Language Models (LLMs).

## `llm_core.llm.base.LLMBase`

This is the abstract base class for all LLM providers. It defines the common interface and underlying logic for model interactions.

**Key Constructor Parameters:**

*   `name (str)`: The specific model identifier to be used (e.g., "gpt-4o-mini", "claude-3-5-sonnet-20240620"). Default: `"model-name"`.
*   `system_prompt (str)`: The system prompt to guide the LLM's behavior. Default: `"You are a helpful assistant"`.
*   `create_completion (Callable)`: A callable (function) that implements the provider-specific logic for making an API call to the LLM. This is typically set by subclasses.
*   `loader (Callable)`: Optional. For models that require explicit loading (like GGUF models via `OpenWeightsModel`), this callable handles the loading process.
*   `loader_kwargs (dict)`: Optional. A dictionary of keyword arguments to pass to the `loader` callable.

**Methods and Properties:**

*   `ask(prompt: str, history: tuple = (), schema: dict = None, temperature: float = 0, tools: list = None, raw_tool_results: bool = False, image_b64: str = None) -> ChatCompletion`:
    *   The primary method for interacting with the LLM.
    *   **Parameters:**
        *   `prompt (str)`: The main user prompt.
        *   `history (tuple)`: A tuple of previous messages in the conversation, where each message is a dict (e.g., `{"role": "user", "content": "..."}`).
        *   `schema (dict)`: An optional JSON schema. If provided, the LLM is instructed to respond in a way that conforms to this schema (often by wrapping it as a function/tool named "PublishAnswer").
        *   `temperature (float)`: The sampling temperature for the LLM (0 for deterministic, higher for more random).
        *   `tools (list)`: A list of tool definitions (dataclasses with a `__call__` method) that the LLM can use.
        *   `raw_tool_results (bool)`: If `True` and a tool is called, the method may return the direct result from the tool (or `None` if the LLM decides not to use a tool when it could have) instead of synthesizing a final response. This is useful if you want to handle tool results externally.
        *   `image_b64 (str)`: Optional base64 encoded string of an image for multimodal LLMs.
    *   **Returns:** A `ChatCompletion` object containing the LLM's response.
    *   **Internal Logic:**
        1.  If `tools` are provided:
            *   The LLM decides which tool (if any) to use from the provided `tools` based on the prompt.
            *   If a tool is selected by the LLM, the tool's `__call__` method is executed with arguments provided by the LLM.
            *   The tool's output is then provided back to the LLM (unless `raw_tool_results` changes this flow) to generate a final user-facing response.
        2.  If `schema` is provided (and no tools take precedence for structured output), it's typically presented to the LLM as a specific tool/function (often named "PublishAnswer" or similar by the framework) to guide its output format according to the schema.
        3.  Calls the provider-specific `create_completion` function to get the LLM's response.
*   `__enter__()` and `__exit__(exc_type, exc_val, exc_tb)`:
    *   Support for using the model as a context manager. Specific model implementations (like `OpenWeightsModel`) might use these to load and release resources. For API-based models, these might be no-ops but ensure consistent usage patterns. *(Self-correction: `LLMBase` itself has them as no-ops. Subclasses like `OpenWeightsModel` override them for actual resource management. The `OpenAIChatModel` and other API models use them for session management if applicable or keep them as no-ops.)*
*   `ctx_size (property)`:
    *   Gets or sets the context window size (in tokens) for the model.
    *   The `ask` method uses `sanitize_prompt` to check if the prompt length exceeds `ctx_size`.
*   `sanitize_prompt(prompt, history=None, schema=None)`:
    *   Estimates the token count of the full prompt (including system prompt, history, and schema) using `tiktoken`.
    *   Raises an `OverflowError` if the estimated token count exceeds `ctx_size`.
    *   Returns the remaining available tokens in the context window.

*(Note: `load_model()` and `release_model()` methods are not explicitly part of `LLMBase`'s public interface but are handled by specific subclasses, often via the `__enter__` and `__exit__` context manager methods, particularly for `OpenWeightsModel`.)*

---

## Provider-Specific Model Classes

These classes inherit from `LLMBase` and configure it for a specific LLM provider. They primarily set the `create_completion` callable and may define default model names or provider-specific parameters.

*   **`llm_core.llm.openai.OpenAIChatModel(name: str = "gpt-4o-mini", ...)`**
    *   For OpenAI models.
    *   Default model: `"gpt-4o-mini"`.
*   **`llm_core.llm.openai.AzureOpenAIChatModel(name: str = "gpt-4o-mini", ...)`**
    *   For Azure OpenAI service models.
    *   Default model: `"gpt-4o-mini"`. Requires Azure-specific environment variables for endpoint and API key.
*   **`llm_core.llm.mistralai.MistralAIModel(name: str = "open-mistral-nemo", ...)`**
    *   For MistralAI models.
    *   Default model: `"open-mistral-nemo"`.
*   **`llm_core.llm.anthropic.AnthropicModel(name: str = "claude-3-5-sonnet-20240620", ...)`**
    *   For Anthropic models.
    *   Default model: `"claude-3-5-sonnet-20240620"`.
*   **`llm_core.llm.google.GoogleAIModel(name: str = "gemini-1.5-flash", ...)`**
    *   For Google AI (Gemini) models.
    *   Default model: `"gemini-1.5-flash"`.
*   **`llm_core.llm.open_weights.OpenWeightsModel(name: str, loader: Callable = llama_cpp_loader, loader_kwargs: dict = None, ...)`**
    *   For GGUF-format open-weight models using `llama-cpp-python`.
    *   `name` usually refers to the local file name (without extension) of the model in the cache directory.
    *   `loader`: Defaults to `llama_cpp_loader` which loads a GGUF model using `llama_cpp.Llama`.
    *   `loader_kwargs`: Passed to the `Llama` constructor (e.g., `{"n_ctx": 4096, "n_gpu_layers": -1}`).
    *   This class makes significant use of `__enter__` (to load the model) and `__exit__` (to free resources).

For all these classes, refer to `LLMBase` for common parameters and methods like `ask()`. Ensure you have the correct API keys set as environment variables for cloud-based models. For `OpenWeightsModel`, ensure models are downloaded to the cache directory (default `~/.cache/py-llm-core/models`).

---

## Response Dataclasses (`llm_core.llm.base`)

The `ask()` method of model classes returns a `ChatCompletion` object, which is composed of several other dataclasses. These provide a structured way to access the LLM's response.

*   **`ChatCompletion`**:
    *   `id (str)`: Unique identifier for the completion.
    *   `model (str)`: Model name used for the completion.
    *   `usage (Usage)`: Token usage statistics.
    *   `object (str)`: Object type (e.g., "chat.completion").
    *   `created (int)`: Timestamp of creation.
    *   `choices (List[ChatCompletionChoice])`: A list of completion choices (usually one).
    *   `system_fingerprint (str)`: Optional. A system fingerprint from the LLM provider.
    *   `prompt_filter_results (dict)`: Optional. Information about content filtering.
    *   `@classmethod parse(attrs: dict)`: Parses a raw dictionary (e.g., from an API JSON response) into a `ChatCompletion` instance. Handles variations between different LLM provider response structures.

*   **`ChatCompletionChoice`**:
    *   `index (int)`: The index of this choice in the list.
    *   `message (Message)`: The message object containing the content.
    *   `finish_reason (str)`: Reason the LLM stopped generating (e.g., "stop", "tool_calls", "length").
    *   `@classmethod from_iterable(cls, iterable)`: Helper to create a list of choices.


*   **`Message`**:
    *   `role (str)`: The role of the message author (e.g., "assistant", "user", "system").
    *   `content (str)`: The textual content of the message. If a tool was called and its output is being processed, this might contain the arguments for the tool call.
    *   `function_call (dict)`: Optional. If the LLM decided to call a function (older OpenAI style).
    *   `tool_calls (list)`: Optional. A list of tool calls requested by the LLM. Each item details the function name and arguments.
    *   `name (str)`: Optional. The name of the function/tool if `role` is "tool".

*   **`Usage`**:
    *   `prompt_tokens (int)`: Number of tokens in the prompt.
    *   `completion_tokens (int)`: Number of tokens in the generated completion.
    *   `total_tokens (int)`: Total tokens used.
    *   `completion_tokens_details (dict)`: Optional. More detailed token information if provided by the API.

These dataclasses aim to provide a somewhat consistent structure over potentially varied raw responses from different LLM providers. The `ChatCompletion.parse` method contains logic to map different provider formats to these fields.
