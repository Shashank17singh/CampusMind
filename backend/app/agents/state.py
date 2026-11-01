"""Shared types for the agent graph."""

import operator
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

from app.llm import ChatMessage

Intent = Literal["qa", "study_plan", "reminder", "summarize", "smalltalk", "out_of_scope"]
INTENTS: tuple[Intent, ...] = (
    "qa",
    "study_plan",
    "reminder",
    "summarize",
    "smalltalk",
    "out_of_scope",
)


@dataclass(frozen=True)
class UserContext:
    """The parts of the signed-in user that agents need."""

    id: uuid.UUID
    role: str = "student"
    department: str | None = None
    semester: int | None = None
    timezone: str = "Asia/Kolkata"


@dataclass
class HandlerResult:
    """What an agent produced."""

    text: str
    data: dict[str, Any] = field(default_factory=dict)
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass(frozen=True)
class HandlerInput:
    """Everything an agent receives. `previous` holds earlier steps' results."""

    message: str
    history: tuple[ChatMessage, ...]
    user: UserContext
    previous: tuple[dict[str, Any], ...]
    emit: Callable[[str], None]


Handler = Callable[[HandlerInput], HandlerResult]


class AgentState(TypedDict, total=False):
    """State passed between graph nodes."""

    messages: list[ChatMessage]
    user: UserContext
    intent: Intent
    confidence: float
    queue: list[Intent]
    tool_results: Annotated[list[dict[str, Any]], operator.add]
    final_answer: str
    error: str | None
