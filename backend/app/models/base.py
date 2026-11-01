"""Declarative base, shared column helpers and enum value sets."""

import uuid
from datetime import UTC, datetime
from typing import Annotated

from sqlalchemy import CheckConstraint, DateTime, MetaData
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, mapped_column

USER_ROLES = ("student", "faculty", "admin")
DOC_TYPES = ("syllabus", "circular", "notice", "regulation", "notes", "contact_list")
DOC_STATUSES = ("uploaded", "extracting", "chunking", "embedding", "ready", "failed")
MESSAGE_ROLES = ("user", "assistant", "system")
FEEDBACK_REASONS = ("wrong", "outdated", "incomplete", "not_relevant")
REMINDER_STATUSES = ("pending", "done", "cancelled")
PLAN_ITEM_TYPES = ("learn", "revise", "practice")
PLAN_ITEM_STATUSES = ("pending", "done", "missed")
SUMMARY_MODES = ("brief", "detailed", "exam_notes", "flashcards")


def utcnow() -> datetime:
    """Return the current time as an aware UTC datetime."""
    return datetime.now(UTC)


def in_values(column: str, values: tuple[str, ...], name: str) -> CheckConstraint:
    """Build a CHECK constraint restricting a text column to the given values."""
    quoted = ", ".join(f"'{value}'" for value in values)
    return CheckConstraint(f"{column} IN ({quoted})", name=name)


UuidPk = Annotated[
    uuid.UUID, mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
]
CreatedAt = Annotated[
    datetime, mapped_column(DateTime(timezone=True), default=utcnow, nullable=False)
]
UpdatedAt = Annotated[
    datetime,
    mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False),
]


class Base(DeclarativeBase):
    """Base class with a naming convention so Alembic produces stable constraint names."""

    metadata = MetaData(
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_N_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )
