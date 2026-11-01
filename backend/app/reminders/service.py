"""Reminder and assignment operations, and the reminder agent."""

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from app.agents.state import Handler, HandlerInput, HandlerResult, UserContext
from app.agents.tools import Tool, ToolRegistry
from app.llm import LLMClient
from app.models import Reminder, Task, User
from app.models.base import utcnow
from app.reminders.parsing import (
    DEFAULT_REMIND_BEFORE,
    Clarification,
    ParsedReminder,
    parse_reminder,
)
from app.reminders.scheduler import RECURRENCE_STEPS

View = Literal["upcoming", "overdue", "done"]

_LIST_REQUEST = re.compile(
    r"\b(what|which|show|list|any|do i have)\b.*\b(deadlines?|reminders?)\b", re.I
)


class ReminderError(Exception):
    """A reminder operation was rejected. The message is safe to show to the user."""


class ReminderArgs(BaseModel):
    """Arguments for the create_reminder tool and the create endpoint."""

    title: str = Field(min_length=1, max_length=300)
    due_at: datetime
    subject: str | None = Field(default=None, max_length=200)
    remind_before: list[int] = Field(default_factory=lambda: list(DEFAULT_REMIND_BEFORE))
    recurrence: Literal["weekly", "daily"] | None = None

    @field_validator("due_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        """A deadline without a timezone is ambiguous, so it is refused."""
        if value.tzinfo is None:
            raise ValueError("due_at must include a timezone")
        return value

    @field_validator("remind_before")
    @classmethod
    def _check_lead_times(cls, value: list[int]) -> list[int]:
        """Lead times are positive and at most 30 days."""
        if len(value) > 5 or any(not 0 < minutes <= 43_200 for minutes in value):
            raise ValueError("Use up to 5 lead times, each between 1 minute and 30 days")
        return value


def create_reminder(
    session: Session,
    user_id: uuid.UUID,
    args: ReminderArgs,
    source_text: str | None = None,
    now: datetime | None = None,
) -> Reminder:
    """Store a reminder. Past dates are refused."""
    if args.due_at <= (now or utcnow()):
        raise ReminderError("That date has already passed.")
    reminder = Reminder(
        user_id=user_id,
        title=args.title,
        subject=args.subject,
        due_at=args.due_at,
        remind_before=args.remind_before,
        recurrence=args.recurrence,
        source_text=source_text,
    )
    session.add(reminder)
    session.flush()
    return reminder


def list_reminders(
    session: Session,
    user_id: uuid.UUID,
    view: View,
    limit: int = 50,
    offset: int = 0,
    now: datetime | None = None,
) -> tuple[list[Reminder], int]:
    """Return one tab of a user's reminders, and how many it holds."""
    moment = now or utcnow()
    filters = [Reminder.user_id == user_id]
    if view == "done":
        filters.append(Reminder.status == "done")
    elif view == "overdue":
        filters += [Reminder.status == "pending", Reminder.due_at < moment]
    else:
        filters += [Reminder.status == "pending", Reminder.due_at >= moment]

    order = Reminder.due_at.desc() if view == "done" else Reminder.due_at
    total = session.scalar(select(func.count()).select_from(Reminder).where(*filters)) or 0
    rows = session.scalars(
        select(Reminder).where(*filters).order_by(order).limit(limit).offset(offset)
    ).all()
    return list(rows), total


def mark_done(reminder: Reminder, now: datetime | None = None) -> None:
    """Finish a reminder. A repeating one moves on to its next occurrence instead."""
    if reminder.recurrence is None:
        reminder.status = "done"
        return
    moment = now or utcnow()
    step = RECURRENCE_STEPS[reminder.recurrence]
    reminder.due_at += step
    while reminder.due_at <= moment:
        reminder.due_at += step
    reminder.snoozed_until = None


def snooze(reminder: Reminder, minutes: int, now: datetime | None = None) -> None:
    """Delay the next alert. A finished reminder cannot be snoozed."""
    if reminder.status != "pending":
        raise ReminderError("Only a pending reminder can be snoozed.")
    reminder.snoozed_until = (now or utcnow()) + timedelta(minutes=minutes)


class AssignmentArgs(BaseModel):
    """A faculty assignment."""

    title: str = Field(min_length=1, max_length=300)
    description: str | None = Field(default=None, max_length=5000)
    subject: str | None = Field(default=None, max_length=200)
    department: str = Field(min_length=1, max_length=50)
    semester: int = Field(ge=1, le=8)
    section: str | None = Field(default=None, max_length=20)
    due_at: datetime

    @field_validator("due_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        """A deadline without a timezone is ambiguous, so it is refused."""
        if value.tzinfo is None:
            raise ValueError("due_at must include a timezone")
        return value


def post_assignment(
    session: Session, faculty: User, args: AssignmentArgs, now: datetime | None = None
) -> tuple[Task, int]:
    """Create an assignment and a reminder for every student in the class."""
    if args.due_at <= (now or utcnow()):
        raise ReminderError("That date has already passed.")
    task = Task(created_by=faculty.id, **args.model_dump())
    session.add(task)
    session.flush()

    filters = [
        User.role == "student",
        User.is_active.is_(True),
        User.department == args.department,
        User.semester == args.semester,
    ]
    if args.section:
        filters.append(User.section == args.section)
    student_ids = session.scalars(select(User.id).where(*filters)).all()
    if student_ids:
        session.execute(
            insert(Reminder)
            .values(
                [
                    {
                        "id": uuid.uuid4(),
                        "user_id": student_id,
                        "task_id": task.id,
                        "title": args.title,
                        "subject": args.subject,
                        "due_at": args.due_at,
                        "remind_before": list(DEFAULT_REMIND_BEFORE),
                    }
                    for student_id in student_ids
                ]
            )
            .on_conflict_do_nothing()
        )
    return task, len(student_ids)


def cancel_assignment(session: Session, task: Task) -> None:
    """Soft-delete an assignment and stop its reminders, including unsent alerts."""
    task.deleted_at = utcnow()
    session.execute(
        update(Reminder)
        .where(Reminder.task_id == task.id, Reminder.status == "pending")
        .values(status="cancelled", updated_at=utcnow())
    )


@dataclass
class ReminderAgent:
    """The chat-facing agent."""

    llm: LLMClient
    session_factory: sessionmaker[Session]

    def _list(self, user: UserContext) -> str:
        """Describe the user's upcoming reminders."""
        with self.session_factory() as session:
            rows, total = list_reminders(session, user.id, "upcoming", limit=10)
        if not rows:
            return "You have no upcoming reminders."
        zone = ZoneInfo(user.timezone)
        lines = [
            f"- {row.title}: {row.due_at.astimezone(zone).strftime('%a %d %b, %I:%M %p')}"
            for row in rows
        ]
        return f"You have {total} upcoming:\n" + "\n".join(lines)

    def _create(self, user: UserContext, source_text: str) -> Tool[ReminderArgs]:
        """Build the tool, bound to the signed-in user so an LLM cannot choose another."""

        def create(args: ReminderArgs) -> str:
            with self.session_factory() as session:
                reminder = create_reminder(session, user.id, args, source_text)
                session.commit()
            due = reminder.due_at.astimezone(ZoneInfo(user.timezone))
            when = due.strftime("%a %d %b, %I:%M %p")
            return f"Done. I'll remind you: {reminder.title}, due {when}."

        return Tool("create_reminder", "Create a reminder", ReminderArgs, create)

    def handle(self, request: HandlerInput) -> HandlerResult:
        """List reminders, or create one, asking for clarification if needed."""
        if _LIST_REQUEST.search(request.message):
            text = self._list(request.user)
        else:
            parsed = parse_reminder(request.message, self.llm, utcnow(), request.user.timezone)
            if isinstance(parsed, Clarification):
                text = parsed.question
            else:
                text = self._save(request.user, parsed, request.message)
        request.emit(text)
        return HandlerResult(text)

    def _save(self, user: UserContext, parsed: ParsedReminder, source_text: str) -> str:
        """Create the reminder through the validated tool."""
        registry = ToolRegistry([self._create(user, source_text)])
        return registry.call(
            "create_reminder",
            {
                "title": parsed.title,
                "due_at": parsed.due_at,
                "subject": parsed.subject,
                "remind_before": list(parsed.remind_before),
                "recurrence": parsed.recurrence,
            },
        )


def make_reminder_handler(llm: LLMClient, session_factory: sessionmaker[Session]) -> Handler:
    """Build the graph handler for the reminder intent."""
    return ReminderAgent(llm, session_factory).handle
