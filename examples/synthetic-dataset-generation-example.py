import json
from typing import List
from enum import Enum
from dataclasses import dataclass, asdict
from llm_core.assistants import LLaMACPPAssistant


class CRUDOperation(Enum):
    CREATE = 0
    READ = 1
    UPDATE = 2
    DELETE = 3


@dataclass
class UserQueryGenerator:
    system_prompt = "You are a helpful assistant."
    prompt = """
    # Goals

    We are developing a new b2b software able to understand user's queries
    written in plain english.

    # Examples of queries

    Cancel all my meetings of the week
    What is on the agenda for the meeting at 1 pm ?
    Show me the log entries for the last 24h
    Disable all access for john.appleseed@gmail.com
    {examples}

    # Todo

    Write {queries_count} new examples of what users could ask.

    """
    user_queries: List[str]

    @classmethod
    def generate(cls, queries_count=10, examples=()):
        with LLaMACPPAssistant(
            cls,
            model="mistral",
            completion_kwargs={"temperature": 0.8},
        ) as assistant:
            batch = assistant.process(
                queries_count=queries_count, examples=examples
            )
            return batch.user_queries


@dataclass
class UserQueryGeneratorV2:
    system_prompt = "You are a helpful assistant."
    prompt = """
    # Goals

    We are developing a new business software able to understand user's commands
    written in plain english. Then we'll map these commands to simple operations
    like create, read, update and delete.

    # Todo

    Imagine {queries_count} examples of what users could ask.

    """
    user_prompts: List[str]

    @classmethod
    def generate(cls, queries_count=20, examples=()):
        with LLaMACPPAssistant(
            cls,
            model="mistral",
            completion_kwargs={"temperature": 1.0},
        ) as assistant:
            batch = assistant.process(queries_count=queries_count)
            return batch.user_prompts


@dataclass
class UserQueryClassification:
    system_prompt = "You are a helpful assistant."
    prompt = """
    # Instructions

    Analyze the user's query and classify his intent into:

    - an operation (among CRUD)
    - a target entity

    # Examples

    query: Cancel all my meetings of the week
    operation: DELETE
    item: MEETING

    query: Show me the log entries for the last 24h
    operation: READ
    item: LOG

    # Todo

    Classify the query: {prompt}
    """
    operation: CRUDOperation
    item: str

    def to_json(self):
        attrs = asdict(self)
        attrs["operation"] = attrs["operation"].name
        return json.dumps(attrs)

    @classmethod
    def classify(cls, prompt):
        with LLaMACPPAssistant(cls, model="mistral") as assistant:
            user_query = assistant.process(prompt=prompt)
            return user_query


def main():
    dataset_queries = []

    with open("dataset.txt", "a") as file:
        for i in range(5):
            queries = UserQueryGeneratorV2.generate()
            for query in queries:
                print(query, file=file, flush=True)
                dataset_queries.append(query)

    with open("dataset.jsonl", "a") as file:
        for query in dataset_queries:
            classification = UserQueryClassification.classify(query)
            print(classification.to_json(), file=file, flush=True)
