from dataclasses import dataclass, field
from llm_core.parsers import NuExtractParser


@dataclass
class Author:
    name: str = ""


@dataclass
class Model:
    name: str = ""
    size: str = ""
    link: str = ""


@dataclass
class Post:
    models: list[Model] = field(default_factory=lambda: [Model()])
    authors: list[Author] = field(default_factory=lambda: [Author()])


# Original LinkedIn Post about the release
# useful to have data *not* in the training set.
text = """Hi Everyone!

So excited to release NuExtract! A big project for us. This is an LLM specialized for the task of structured extraction. You can use it to transform any kind of text into a structured output by only giving it a template, and thus tackle any information extraction task you have. It is open-source (MIT licence) and available on Hugging Face.

NuExtract is small (3.8B) and reaches near GPT-4 level on zero-shot structured extraction tasks 😃 You can try it here: https://lnkd.in/emtNpNu3

We trained three version of this model:
- NuExtract-tiny (0.5B) - https://lnkd.in/eMcVSPNr
- NuExtract (3.8B) - https://lnkd.in/eQ-9zaWW
- NuExtract-large (7B) - https://lnkd.in/e5b73Hc5

To create these models we basically took Llama-3 70B to annotate 50k documents and fine tuned Phi-3-mini, Phi-3-small, and Qwen1.5-0.5B on it. We had to figure out lots of interesting stuff during this project, such as finding a way to fight hallucinations. You can read the full story here: https://lnkd.in/eqxjyQza

Congrats to Alexandre Constantin for this beautiful project, to Sergei Bogdanov 💫 for helping out in the beginning, and to Liam Cripwell for working on the evaluation. The path is set to solve information extraction 😃!
"""

with NuExtractParser(Post) as parser:
    response = parser.parse(text)
    print(response)
