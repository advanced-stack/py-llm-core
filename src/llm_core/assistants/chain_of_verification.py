# -*- coding: utf-8 -*-
from typing import List
from dataclasses import dataclass

from .base import OpenAIAssistant


@dataclass
class Baseline:
    system_prompt = "You are a knowledgeable and helpful assistant"

    prompt = """
    {instructions}
    {question}
    """

    answer: str

    @classmethod
    def ask(cls, question, content="", model="gpt-3.5-turbo"):
        if content:
            instructions = "\n".join(
                (
                    f"Content: {content}",
                    "Using only the previous content, answer the following:",
                )
            )
        else:
            instructions = "Answer the following:"

        with OpenAIAssistant(cls, model) as assistant:
            return assistant.process(
                instructions=instructions, question=question
            )


@dataclass
class BaselineVerification:
    system_prompt = "You are a helpful assistant"

    prompt = """
    {instructions}

    The answer was:
    {answer}


    Write {n_questions} single-fact, close-ended questions to help verify if there are
    mistakes in the answer.
    """

    questions: List[str]

    @classmethod
    def control(
        cls,
        question,
        answer,
        content="",
        n_questions=10,
        model="gpt-3.5-turbo",
    ):
        if content:
            instructions = "\n".join(
                (
                    f"Content: {content}",
                    "---- We asked the following:"
                    "Using only the previous content, answer the following query:",
                    question,
                )
            )
        else:
            instructions = f"We asked the following query: {question}"

        with OpenAIAssistant(cls, model) as assistant:
            return assistant.process(
                instructions=instructions,
                answer=answer,
                n_questions=n_questions,
            )


@dataclass
class ExecuteVerification:
    system_prompt = """
    You are a meticulous assistant: You answer only when you are sure.
    Don't start your sentences by Yes or No. Use a neutral tone.
    """

    prompt = """
    {instructions}
    {question}
    """

    answer: str

    @classmethod
    def from_iterable(cls, questions, content="", model="gpt-3.5-turbo"):
        for question in questions:
            yield cls.ask(question, content).answer

    @classmethod
    def ask(cls, question, content="", model="gpt-3.5-turbo"):
        if content:
            instructions = "\n".join(
                (
                    f"Content: {content}",
                    "Using only the previous content:",
                )
            )
        else:
            instructions = ""

        with OpenAIAssistant(cls, model) as assistant:
            return assistant.process(
                instructions=instructions, question=question
            )


@dataclass
class RevisedBaseline:
    system_prompt = """You are meticulous assistant and you need to
    be sure of your answers before responding"""

    prompt = """
    We asked the following:
    {content}
    > {question}

    {baseline}

    --
    A second source gave us the following facts:
    {qa_pairs}


    Identifying inconsistencies between the first source and the second source,
    try to provide a definitive answer for:
    {question}
    """

    answer: str

    @classmethod
    def revise(
        cls, question, baseline, qa_pairs, content="", model="gpt-3.5-turbo"
    ):
        with OpenAIAssistant(cls, model) as assistant:
            return assistant.process(
                question=question,
                baseline=baseline,
                content=content,
                qa_pairs=qa_pairs,
            )


@dataclass
class COVQuestionAnswering:
    question: str
    baseline_answer: str
    verification_questions: List[str]
    verification_answers: List[str]
    revised_answer: str
    content: str = ""

    @classmethod
    def ask(cls, question, content="", n_questions=10, model="gpt-3.5-turbo"):
        baseline = Baseline.ask(
            question=question, content=content, model=model
        )

        verification = BaselineVerification.control(
            question=question,
            answer=baseline.answer,
            content=content,
            n_questions=n_questions,
            model=model,
        )

        verification_answers = list(
            ExecuteVerification.from_iterable(
                questions=verification.questions, content=content, model=model
            )
        )
        qa_pairs = zip(
            verification.questions,
            verification_answers,
        )
        qa_results = "\n >".join(verification_answers)
        # qa_results = "\n".join("Q:{}\nA:{}".format(q, a) for q, a in qa_pairs)

        revised_answer = RevisedBaseline.revise(
            question=question,
            baseline=baseline.answer,
            qa_pairs=qa_results,
            content=content,
            model=model,
        ).answer

        return cls(
            question,
            baseline.answer,
            verification.questions,
            verification_answers,
            revised_answer,
            content,
        )
