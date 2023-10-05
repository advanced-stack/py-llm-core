# -*- coding: utf-8 -*-
import requests
from llm_core.splitters import TokenSplitter
from llm_core.assistants import (
    Analyst,
    Doubter,
    ConsistencyVerifier,
    LLaMACPPAssistant,
)

# Fetch some content
pizza_dough_recipe_url = (
    "https://raw.githubusercontent.com/hendricius/pizza-dough/main/README.md"
)

model = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
assistant_cls = LLaMACPPAssistant

# Utilities
analyst = Analyst(model, assistant_cls)
doubter = Doubter(model, assistant_cls)
verifier = ConsistencyVerifier(model, assistant_cls)

splitter = TokenSplitter(model=model, chunk_size=3_000)
pizza_dough_recipe = requests.get(pizza_dough_recipe_url).text
context = splitter.first_extract(pizza_dough_recipe)

query = "Write 3 advices when making pizza dough."


analyst_response = analyst.ask(query, context)
print(analyst_response.content)

question_collection = doubter.verify(query, analyst_response.content)
questions = question_collection.questions
print(questions)


answers = []

for question in questions:
    response = analyst.ask(question, context=context)
    answers.append(response.content)

for question, answer in zip(questions, answers):
    verifications = verifier.verify(
        question=question, context=context, answer=response.content
    )
    print(question, answer, verifications)
