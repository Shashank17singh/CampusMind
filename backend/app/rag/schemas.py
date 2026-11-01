"""Input and output types for the RAG pipeline."""

import uuid
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from app.llm import ChatMessage

NOT_FOUND = "I couldn't find this in the official documents"

Confidence = Literal["high", "medium", "low"]


class Citation(BaseModel):
    """A source used in an answer. `number` matches the [n] marker in the answer text."""

    number: int
    document_id: uuid.UUID
    title: str
    page: int | None
    snippet: str


class RagAnswer(BaseModel):
    """The final answer."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence = "low"
    suggested_followups: list[str] = Field(default_factory=list, max_length=3)
    answered: bool = False
    standalone_question: str = ""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False


@dataclass(frozen=True)
class UserProfile:
    """The parts of a user that shape retrieval."""

    department: str | None = None
    semester: int | None = None


@dataclass(frozen=True)
class RagRequest:
    """A question and its context."""

    question: str
    profile: UserProfile = field(default_factory=UserProfile)
    history: tuple[ChatMessage, ...] = ()


@dataclass(frozen=True)
class TokenEvent:
    """A piece of the answer text."""

    text: str


@dataclass(frozen=True)
class FinalEvent:
    """The finished answer, sent after the last token."""

    answer: RagAnswer
