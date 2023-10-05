# -*- coding: utf-8 -*-
from typing import List
from dataclasses import dataclass


@dataclass
class QuestionCollection:
    system_prompt = "You are a helpful assistant."

    prompt = """Instructions:
    ```
    {instructions}
    ```

    The provided answer was:
    ```
    {answer}
    ```

    Write {n_questions} single-fact, close-ended questions to help verify
    if there are mistakes in the answer.
    """

    questions: List[str]


@dataclass
class Doubter:
    model: str
    assistant_cls: type
    results_cls: type = QuestionCollection

    def verify(self, instructions, answer, n_questions=5):
        with self.assistant_cls(
            self.results_cls, model=self.model
        ) as assistant:
            verification = assistant.process(
                instructions=instructions,
                answer=answer,
                n_questions=n_questions,
            )
            return verification


@dataclass
class AnswerConsistency:
    system_prompt = (
        "You are a meticulous assistant checking for inconsistencies"
    )

    prompt = """Context:
    ```
    {context}
    ```

    Question:
    ```
    {question}
    ```

    The provided Answer was:
    ```
    {answer}
    ```

    - Does the Answer is consistent ?
    - Does the Answer is inferred from the Context ?
    """

    is_consistent: bool
    is_inferred_from_context: bool


@dataclass
class ConsistencyVerifier:
    model: str
    assistant_cls: type
    results_cls: type = AnswerConsistency

    def verify(self, question, context, answer):
        with self.assistant_cls(
            self.results_cls, model=self.model
        ) as assistant:
            verification = assistant.process(
                question=question,
                context=context,
                answer=answer,
            )
            return verification
