# -*- coding: utf-8 -*-
from dataclasses import dataclass


@dataclass
class Answer:
    system_prompt = "You are a helpful analyst."

    prompt = """Context:
    ```
    {context}
    ```

    Using *only* the Context, answer briefly to the following:
    ```
    {question}
    ```
    """

    content: str


@dataclass
class Analyst:
    model: str
    assistant_cls: type
    results_cls: type = Answer

    def batch_ask(self, questions, context):
        for question in questions:
            with self.assistant_cls(
                self.results_cls, model=self.model
            ) as assistant:
                answer = assistant.process(question=question, context=context)

            # Release before yielding
            yield answer

    def ask(self, question, context):
        with self.assistant_cls(
            self.results_cls, model=self.model
        ) as assistant:
            answer = assistant.process(question=question, context=context)
            return answer
