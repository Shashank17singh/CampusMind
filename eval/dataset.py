"""The labelled test set: its format, and the rule that only verified questions count."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Category = Literal["factual", "multi-hop", "unanswerable", "time-sensitive"]
TARGET_SIZE = 150


class LabelledQuestion(BaseModel):
    """One labelled question."""

    question: str = Field(min_length=3)
    gold_answer: str
    relevant_document_ids: list[str] = Field(default_factory=list)
    category: Category
    verified: bool = False
    synthetic: bool = False

    @model_validator(mode="after")
    def _check_consistency(self) -> "LabelledQuestion":
        """An unanswerable question has no source. Every other kind must have one."""
        if self.category == "unanswerable" and self.relevant_document_ids:
            raise ValueError("An unanswerable question cannot have relevant documents")
        if self.category != "unanswerable" and not self.relevant_document_ids:
            raise ValueError(f"A {self.category} question needs relevant documents")
        return self


class UnverifiedDataError(Exception):
    """The test set has not been checked by a person."""


def load_testset(path: Path, require_verified: bool = True) -> list[LabelledQuestion]:
    """Read and validate a test set.

    LLM-drafted questions are not ground truth. Any that a person has not verified stop
    the run, so an unchecked draft cannot end up behind a number in the paper.
    """
    questions = [
        LabelledQuestion.model_validate(item) for item in json.loads(path.read_text("utf-8"))
    ]
    unverified = sum(1 for question in questions if not question.verified)
    if require_verified and unverified:
        raise UnverifiedDataError(
            f"{unverified} of {len(questions)} questions are unverified. Review them, set "
            '"verified": true, or pass --allow-unverified for a trial run.'
        )
    return questions


def save_testset(questions: list[LabelledQuestion], path: Path) -> None:
    """Write a test set as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [question.model_dump() for question in questions]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def summarize(questions: list[LabelledQuestion]) -> dict[str, int]:
    """Count questions per category."""
    counts: dict[str, int] = {}
    for question in questions:
        counts[question.category] = counts.get(question.category, 0) + 1
    return counts
