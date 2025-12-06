from __future__ import annotations

from typing import Literal, TypedDict

QuestionType = Literal["MCQ", "NUM"]


class BaseQuestion(TypedDict):
    index: int
    text: str  # the statement of the question


class MCQQuestion(BaseQuestion):
    type: Literal["MCQ"]
    choices: list[str]
    correct: list[str]


class NumQuestion(BaseQuestion):
    type: Literal["NUM"]
    correct: float
    tolerance: float


class Exam(TypedDict):
    exam_id: str
    title: str | None
    questions: list[MCQQuestion | NumQuestion]
    variants: int
    variant_ordering: list[list[int]]
