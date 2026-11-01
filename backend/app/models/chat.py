"""Conversations, messages, feedback and agent traces."""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    SmallInteger,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import (
    FEEDBACK_REASONS,
    MESSAGE_ROLES,
    Base,
    CreatedAt,
    UuidPk,
    in_values,
    utcnow,
)


class Conversation(Base):
    """A chat thread owned by one user."""

    __tablename__ = "conversations"

    id: Mapped[UuidPk]
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[CreatedAt]
    last_message_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_conversations_user_recent", "user_id", text("last_message_at DESC")),
        Index("ix_conversations_last_message_at", "last_message_at"),
    )


class Message(Base):
    """One user or assistant turn, with answer metadata on assistant turns."""

    __tablename__ = "messages"

    id: Mapped[UuidPk]
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    intent: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[str | None] = mapped_column(Text)
    answered: Mapped[bool | None] = mapped_column(Boolean)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    tokens_in: Mapped[int | None] = mapped_column(Integer)
    tokens_out: Mapped[int | None] = mapped_column(Integer)
    model_name: Mapped[str | None] = mapped_column(Text)
    trace_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    search_vector: Mapped[Any] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', content)", persisted=True)
    )
    created_at: Mapped[CreatedAt]

    __table_args__ = (
        in_values("role", MESSAGE_ROLES, "role_valid"),
        Index("ix_messages_conversation_created", "conversation_id", "created_at"),
        Index("ix_messages_search_vector", "search_vector", postgresql_using="gin"),
        Index("ix_messages_created_at", "created_at"),
        Index("ix_messages_user_created", "user_id", text("created_at DESC")),
        Index(
            "ix_messages_unanswered",
            "created_at",
            postgresql_where=text("answered IS FALSE"),
        ),
    )


class Feedback(Base):
    """A user's rating of an assistant message."""

    __tablename__ = "feedback"

    id: Mapped[UuidPk]
    message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    reason: Mapped[str | None] = mapped_column(Text)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[CreatedAt]

    __table_args__ = (
        CheckConstraint("rating IN (-1, 1)", name="rating_valid"),
        in_values("reason", FEEDBACK_REASONS, "reason_valid"),
        Index("uq_feedback_message_user", "message_id", "user_id", unique=True),
        Index("ix_feedback_created_at", "created_at"),
    )


class AgentTrace(Base):
    """One graph node's execution within a request."""

    __tablename__ = "agent_traces"

    id: Mapped[int] = mapped_column(BigInteger, Identity(always=True), primary_key=True)
    trace_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    step: Mapped[int] = mapped_column(Integer, nullable=False)
    node: Mapped[str] = mapped_column(Text, nullable=False)
    input: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    output: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )

    __table_args__ = (
        Index("ix_agent_traces_trace_step", "trace_id", "step"),
        Index("ix_agent_traces_created_at", "created_at"),
    )
