# -*- coding: utf-8 -*-
import os
from decouple import config

MODELS_CACHE_DIR = config(
    "MODELS_CACHE_DIR",
    default=os.path.expanduser("~/.cache/py-llm-core/models"),
)

if not os.path.exists(MODELS_CACHE_DIR):
    os.makedirs(MODELS_CACHE_DIR)

#: Store models in the same directory to simplify purges
os.environ["TIKTOKEN_CACHE_DIR"] = MODELS_CACHE_DIR


#: OpenAI
OPENAI_API_KEY = config("OPENAI_API_KEY", default=None)

USE_AZURE_OPENAI = config("USE_AZURE_OPENAI", cast=bool, default=False)

AZURE_OPENAI_ENDPOINT = config("AZURE_OPENAI_ENDPOINT", default=None)


#: Mistral
MISTRAL_API_KEY = config("MISTRAL_API_KEY", default=None)

USE_AZURE_AI_MISTRAL_LARGE = config(
    "USE_AZURE_AI_MISTRAL_LARGE", cast=bool, default=False
)

AZURE_AI_MISTRAL_LARGE_ENDPOINT = config(
    "AZURE_AI_MISTRAL_LARGE_ENDPOINT", default=None
)

AZURE_AI_MISTRAL_LARGE_KEY = config("AZURE_AI_MISTRAL_LARGE_KEY", default=None)


DEFAULT_TIMEOUT = config("DEFAULT_TIMEOUT", default=300.0)
