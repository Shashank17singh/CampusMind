"""Conversation, message and feedback endpoints."""

import json
import time
import uuid
from collections.abc import Iterator
from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import func, select

from app.agents.graph import AgentOutcome, AgentRunner
from app.agents.runner import get_agent_runner
from app.agents.state import UserContext
from app.auth.deps import CurrentUser, DbSession, require_role
from app.config import get_settings
from app.conversations import service
from app.conversations.tasks import auto_title
from app.db import session_scope
from app.llm import ChatMessage
from app.logging import get_logger
from app.models import Conversation, Feedback, Message, User
from app.rag.service import format_sse

router = APIRouter(tags=["conversations"])
log = get_logger(__name__)

Reason = Literal["wrong", "outdated", "incomplete", "not_relevant"]
PageLimit = Annotated[int, Query(ge=1, le=100)]
PageOffset = Annotated[int, Query(ge=0)]


class ConversationCreate(BaseModel):
    """A new conversation."""

    title: str | None = Field(default=None, max_length=200)


class ConversationRename(BaseModel):
    """A new title."""

    title: str = Field(min_length=1, max_length=200)


class ConversationOut(BaseModel):
    """A conversation."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str | None
    created_at: datetime
    last_message_at: datetime


class ConversationPage(BaseModel):
    """One page of conversations."""

    items: list[ConversationOut]
    total: int


class MessageOut(BaseModel):
    """A message, with answer metadata on assistant messages."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    citations: list[dict[str, Any]]
    intent: str | None
    confidence: str | None
    latency_ms: int | None
    tokens_in: int | None
    tokens_out: int | None
    model_name: str | None
    trace_id: uuid.UUID | None
    created_at: datetime


class MessagePage(BaseModel):
    """One page of messages."""

    items: list[MessageOut]
    total: int


class MessageCreate(BaseModel):
    """What the user typed."""

    content: str = Field(min_length=1, max_length=4000)


class SearchHitOut(BaseModel):
    """A message matching a search."""

    message_id: uuid.UUID
    conversation_id: uuid.UUID
    snippet: str
    created_at: datetime


class SearchPage(BaseModel):
    """One page of search results."""

    items: list[SearchHitOut]
    total: int


class FeedbackCreate(BaseModel):
    """A rating. A reason only makes sense on a thumbs down."""

    rating: Literal[1, -1]
    reason: Reason | None = None
    comment: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def _check_reason(self) -> "FeedbackCreate":
        """Thumbs up has no reason."""
        if self.rating == 1 and self.reason is not None:
            raise ValueError("A reason is only used with a thumbs down")
        return self


class FeedbackOut(BaseModel):
    """A stored rating."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    message_id: uuid.UUID
    user_id: uuid.UUID
    rating: int
    reason: str | None
    comment: str | None
    created_at: datetime


class FeedbackPage(BaseModel):
    """One page of ratings."""

    items: list[FeedbackOut]
    total: int


def _owned(db: DbSession, conversation_id: uuid.UUID, user: CurrentUser) -> Conversation:
    """Load the user's conversation. Someone else's looks the same as a missing one."""
    conversation = service.get_owned(db, conversation_id, user.id)
    if conversation is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Conversation not found")
    return conversation


@router.post("/conversations", response_model=ConversationOut, status_code=201)
def create_conversation(body: ConversationCreate, db: DbSession, user: CurrentUser) -> Conversation:
    """Start a conversation."""
    conversation = Conversation(user_id=user.id, title=body.title)
    db.add(conversation)
    db.commit()
    return conversation


@router.get("/conversations", response_model=ConversationPage)
def list_conversations(
    db: DbSession, user: CurrentUser, limit: PageLimit = 20, offset: PageOffset = 0
) -> ConversationPage:
    """List conversations, most recently active first."""
    filters = [Conversation.user_id == user.id]
    total = db.scalar(select(func.count()).select_from(Conversation).where(*filters)) or 0
    rows = db.scalars(
        select(Conversation)
        .where(*filters)
        .order_by(Conversation.last_message_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return ConversationPage(
        items=[ConversationOut.model_validate(row) for row in rows], total=total
    )


@router.get("/conversations/search", response_model=SearchPage)
def search(
    db: DbSession,
    user: CurrentUser,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    limit: PageLimit = 20,
    offset: PageOffset = 0,
) -> SearchPage:
    """Search all of my messages."""
    hits, total = service.search_messages(db, user.id, q, limit, offset)
    return SearchPage(items=[SearchHitOut(**hit.__dict__) for hit in hits], total=total)


@router.get("/conversations/export")
def export(db: DbSession, user: CurrentUser) -> Response:
    """Download my history as a JSON file."""
    return Response(
        content=json.dumps(service.export_history(db, user.id), ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="campusmind-history.json"'},
    )


@router.get("/conversations/{conversation_id}/messages", response_model=MessagePage)
def get_messages(
    conversation_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    offset: PageOffset = 0,
) -> MessagePage:
    """Return a conversation's messages, oldest first."""
    _owned(db, conversation_id, user)
    filters = [Message.conversation_id == conversation_id]
    total = db.scalar(select(func.count()).select_from(Message).where(*filters)) or 0
    rows = db.scalars(
        select(Message).where(*filters).order_by(Message.created_at).limit(limit).offset(offset)
    ).all()
    return MessagePage(items=[MessageOut.model_validate(row) for row in rows], total=total)


@router.patch("/conversations/{conversation_id}", response_model=ConversationOut)
def rename_conversation(
    conversation_id: uuid.UUID, body: ConversationRename, db: DbSession, user: CurrentUser
) -> Conversation:
    """Rename a conversation."""
    conversation = _owned(db, conversation_id, user)
    conversation.title = body.title
    db.commit()
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: uuid.UUID, db: DbSession, user: CurrentUser) -> None:
    """Delete a conversation and its messages."""
    db.delete(_owned(db, conversation_id, user))
    db.commit()


def _citation_event(outcome: AgentOutcome) -> str:
    """Build the citations event from the answering step."""
    details: dict[str, Any] = next(
        (step["rag"] for step in outcome.tool_results if "rag" in step), {}
    )
    return format_sse(
        "citations",
        {
            "citations": details.get("citations", []),
            "confidence": details.get("confidence"),
            "suggested_followups": details.get("suggested_followups", []),
        },
    )


def _stream_turn(
    runner: AgentRunner,
    conversation_id: uuid.UUID,
    user: UserContext,
    history: list[ChatMessage],
    message_id: uuid.UUID,
) -> Iterator[str]:
    """Run one turn as Server-Sent Events, and store the answer however it ends.

    If the client disconnects, which is how the stop button works, this generator is
    closed partway through, and `finally` stores the text produced up to that point. No
    database connection is held during the slow LLM calls.
    """
    trace_id = uuid.uuid4()
    started = time.perf_counter()
    parts: list[str] = []
    finished = False
    yield format_sse("meta", {"trace_id": str(trace_id), "message_id": str(message_id)})
    try:
        for item in runner.stream(user, history, trace_id):
            if isinstance(item, AgentOutcome):
                latency = int((time.perf_counter() - started) * 1000)
                with session_scope() as session:
                    conversation = session.get_one(Conversation, conversation_id)
                    service.save_answer(
                        session,
                        conversation,
                        message_id,
                        item,
                        latency,
                        get_settings().resolved_llm_model,
                    )
                finished = True
                yield _citation_event(item)
                yield format_sse(
                    "done",
                    {
                        "intent": item.intent,
                        "latency_ms": latency,
                        "tokens_in": item.tokens_in,
                        "tokens_out": item.tokens_out,
                    },
                )
            else:
                parts.append(item)
                yield format_sse("token", {"text": item})
    finally:
        if not finished:
            latency = int((time.perf_counter() - started) * 1000)
            log.info("answer_stopped_early", trace_id=str(trace_id), chars=len("".join(parts)))
            with session_scope() as session:
                still_exists = session.get(Conversation, conversation_id)
                if still_exists is not None and parts:
                    service.save_partial_answer(
                        session, still_exists, message_id, "".join(parts), trace_id, latency
                    )


@router.post("/conversations/{conversation_id}/messages")
def send_message(
    conversation_id: uuid.UUID,
    body: MessageCreate,
    db: DbSession,
    user: CurrentUser,
    runner: Annotated[AgentRunner, Depends(get_agent_runner)],
) -> StreamingResponse:
    """Send a message, streaming the assistant's reply."""
    conversation = _owned(db, conversation_id, user)
    is_first = conversation.title is None and not db.scalar(
        select(func.count()).select_from(Message).where(Message.conversation_id == conversation.id)
    )
    service.add_user_message(db, conversation, body.content)
    history = service.recent_history(db, conversation.id)
    db.commit()
    if is_first:
        auto_title.delay(str(conversation.id), body.content)

    context = UserContext(user.id, user.role, user.department, user.semester, user.timezone)
    return StreamingResponse(
        _stream_turn(runner, conversation.id, context, history, uuid.uuid4()),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/messages/{message_id}/feedback", response_model=FeedbackOut)
def give_feedback(
    message_id: uuid.UUID, body: FeedbackCreate, db: DbSession, user: CurrentUser
) -> Feedback:
    """Rate one of my assistant messages."""
    message = db.get(Message, message_id)
    if message is None or message.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Message not found")
    if message.role != "assistant":
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Only assistant messages can be rated"
        )
    feedback = service.save_feedback(db, message, user.id, body.rating, body.reason, body.comment)
    db.commit()
    return feedback


@router.get("/admin/feedback", response_model=FeedbackPage)
def list_feedback(
    db: DbSession,
    _: Annotated[User, Depends(require_role("admin"))],
    rating: Annotated[int | None, Query(ge=-1, le=1)] = None,
    reason: Reason | None = None,
    limit: PageLimit = 50,
    offset: PageOffset = 0,
) -> FeedbackPage:
    """List ratings for the admin dashboard, newest first."""
    filters = []
    if rating is not None:
        filters.append(Feedback.rating == rating)
    if reason is not None:
        filters.append(Feedback.reason == reason)
    total = db.scalar(select(func.count()).select_from(Feedback).where(*filters)) or 0
    rows = db.scalars(
        select(Feedback)
        .where(*filters)
        .order_by(Feedback.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return FeedbackPage(items=[FeedbackOut.model_validate(row) for row in rows], total=total)
