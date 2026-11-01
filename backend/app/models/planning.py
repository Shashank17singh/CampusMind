"""Assignments, reminders, notifications and study plans."""

import uuid
from datetime import date, datetime, time
from typing import Any

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    Text,
    Time,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import (
    PLAN_ITEM_STATUSES,
    PLAN_ITEM_TYPES,
    REMINDER_STATUSES,
    Base,
    CreatedAt,
    UpdatedAt,
    UuidPk,
    in_values,
)


class Task(Base):
    """An assignment posted by faculty for a class."""

    __tablename__ = "tasks"

    id: Mapped[UuidPk]
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL")
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    subject: Mapped[str | None] = mapped_column(Text)
    department: Mapped[str] = mapped_column(Text, nullable=False)
    semester: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    section: Mapped[str | None] = mapped_column(Text)
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[CreatedAt]

    __table_args__ = (Index("ix_tasks_class_due", "department", "semester", "due_at"),)


class Reminder(Base):
    """A user's deadline. Faculty assignments create one per student."""

    __tablename__ = "reminders"

    id: Mapped[UuidPk]
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    task_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tasks.id", ondelete="CASCADE")
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    subject: Mapped[str | None] = mapped_column(Text)
    due_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    remind_before: Mapped[list[int]] = mapped_column(
        ARRAY(Integer), nullable=False, default=lambda: [1440, 120]
    )
    recurrence: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")
    snoozed_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_text: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]

    __table_args__ = (
        in_values("status", REMINDER_STATUSES, "status_valid"),
        Index("ix_reminders_user_status_due", "user_id", "status", "due_at"),
        Index(
            "uq_reminders_task_user",
            "task_id",
            "user_id",
            unique=True,
            postgresql_where=text("task_id IS NOT NULL"),
        ),
    )


class ReminderNotification(Base):
    """One delivery of a reminder on one channel at one fire time."""

    __tablename__ = "reminder_notifications"

    id: Mapped[UuidPk]
    reminder_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("reminders.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    channel: Mapped[str] = mapped_column(Text, nullable=False)
    fire_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(Text, nullable=False)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    read_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    __table_args__ = (
        in_values("channel", ("in_app", "email"), "channel_valid"),
        Index("uq_reminder_notifications_key", "idempotency_key", unique=True),
        Index(
            "ix_reminder_notifications_due",
            "fire_at",
            postgresql_where=text("sent_at IS NULL"),
        ),
        Index("ix_reminder_notifications_user_sent", "user_id", text("sent_at DESC")),
    )


class StudyPlan(Base):
    """A generated study schedule and the inputs that produced it."""

    __tablename__ = "study_plans"

    id: Mapped[UuidPk]
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    inputs: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[CreatedAt]
    updated_at: Mapped[UpdatedAt]

    items: Mapped[list["PlanItem"]] = relationship(
        order_by="(PlanItem.date, PlanItem.start_time)",
        cascade="all, delete-orphan",
        passive_deletes=True,
        viewonly=False,
    )

    __table_args__ = (Index("ix_study_plans_user_active", "user_id", "is_active"),)


class PlanItem(Base):
    """One scheduled slot in a study plan."""

    __tablename__ = "plan_items"

    id: Mapped[UuidPk]
    plan_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("study_plans.id", ondelete="CASCADE"), nullable=False
    )
    date: Mapped[date] = mapped_column(Date, nullable=False)
    start_time: Mapped[time] = mapped_column(Time, nullable=False)
    end_time: Mapped[time] = mapped_column(Time, nullable=False)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, default="pending")

    __table_args__ = (
        in_values("type", PLAN_ITEM_TYPES, "type_valid"),
        in_values("status", PLAN_ITEM_STATUSES, "status_valid"),
        Index("ix_plan_items_plan_date", "plan_id", "date", "start_time"),
    )
