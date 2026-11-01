"""Reminder, assignment and notification endpoints."""

import uuid
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import ColumnElement, func, select

from app.auth.deps import CurrentUser, DbSession, require_role
from app.llm import LLMClient, get_llm
from app.models import Reminder, ReminderNotification, Task, User
from app.models.base import utcnow
from app.reminders.parsing import Clarification, parse_reminder
from app.reminders.service import (
    AssignmentArgs,
    ReminderArgs,
    ReminderError,
    View,
    cancel_assignment,
    create_reminder,
    list_reminders,
    mark_done,
    post_assignment,
    snooze,
)

router = APIRouter(tags=["reminders"])

FacultyUser = Annotated[User, Depends(require_role("faculty", "admin"))]


class ReminderOut(BaseModel):
    """A reminder."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    subject: str | None
    due_at: datetime
    remind_before: list[int]
    recurrence: str | None
    status: str
    snoozed_until: datetime | None
    task_id: uuid.UUID | None


class ReminderPage(BaseModel):
    """One page of reminders."""

    items: list[ReminderOut]
    total: int


class ParseRequest(BaseModel):
    """A reminder in natural language."""

    text: str = Field(min_length=1, max_length=1000)


class ParseResult(BaseModel):
    """Either a created reminder, or a question to ask the user."""

    reminder: ReminderOut | None = None
    clarification: str | None = None


class SnoozeRequest(BaseModel):
    """How long to snooze for."""

    minutes: int = Field(ge=1, le=10_080)


class AssignmentOut(BaseModel):
    """A faculty assignment."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    description: str | None
    subject: str | None
    department: str
    semester: int
    section: str | None
    due_at: datetime
    created_by: uuid.UUID | None
    created_at: datetime
    reminders_created: int | None = None


class AssignmentPage(BaseModel):
    """One page of assignments."""

    items: list[AssignmentOut]
    total: int


class NotificationOut(BaseModel):
    """A delivered in-app notification."""

    id: uuid.UUID
    reminder_id: uuid.UUID
    title: str
    due_at: datetime
    sent_at: datetime
    read_at: datetime | None


class NotificationPage(BaseModel):
    """One page of notifications."""

    items: list[NotificationOut]
    total: int


def _load(db: DbSession, reminder_id: uuid.UUID, user: CurrentUser) -> Reminder:
    """Load the user's own reminder. Someone else's looks the same as a missing one."""
    reminder = db.get(Reminder, reminder_id)
    if reminder is None or reminder.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Reminder not found")
    return reminder


def _reject(exc: ReminderError) -> HTTPException:
    """Convert a rejected operation to a 422."""
    return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc))


@router.post("/reminders", response_model=ReminderOut, status_code=status.HTTP_201_CREATED)
def create(body: ReminderArgs, db: DbSession, user: CurrentUser) -> Reminder:
    """Create a reminder from structured fields."""
    try:
        reminder = create_reminder(db, user.id, body)
    except ReminderError as exc:
        raise _reject(exc) from exc
    db.commit()
    return reminder


@router.post("/reminders/parse", response_model=ParseResult)
def parse(
    body: ParseRequest,
    response: Response,
    db: DbSession,
    user: CurrentUser,
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> ParseResult:
    """Create a reminder from natural language, or return a question if it is unclear."""
    parsed = parse_reminder(body.text, llm, utcnow(), user.timezone)
    if isinstance(parsed, Clarification):
        return ParseResult(clarification=parsed.question)

    args = ReminderArgs(
        title=parsed.title,
        due_at=parsed.due_at,
        subject=parsed.subject,
        remind_before=list(parsed.remind_before),
        recurrence=parsed.recurrence,  # type: ignore[arg-type]
    )
    reminder = create_reminder(db, user.id, args, source_text=body.text)
    db.commit()
    response.status_code = status.HTTP_201_CREATED
    return ParseResult(reminder=ReminderOut.model_validate(reminder))


@router.get("/reminders", response_model=ReminderPage)
def list_all(
    db: DbSession,
    user: CurrentUser,
    view: View = "upcoming",
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> ReminderPage:
    """List the upcoming, overdue or done tab."""
    rows, total = list_reminders(db, user.id, view, limit, offset)
    return ReminderPage(items=[ReminderOut.model_validate(row) for row in rows], total=total)


@router.post("/reminders/{reminder_id}/done", response_model=ReminderOut)
def done(reminder_id: uuid.UUID, db: DbSession, user: CurrentUser) -> Reminder:
    """Mark a reminder done. A repeating one moves to its next occurrence."""
    reminder = _load(db, reminder_id, user)
    mark_done(reminder)
    db.commit()
    return reminder


@router.post("/reminders/{reminder_id}/snooze", response_model=ReminderOut)
def snooze_reminder(
    reminder_id: uuid.UUID, body: SnoozeRequest, db: DbSession, user: CurrentUser
) -> Reminder:
    """Delay a reminder's next alert."""
    reminder = _load(db, reminder_id, user)
    try:
        snooze(reminder, body.minutes)
    except ReminderError as exc:
        raise _reject(exc) from exc
    db.commit()
    return reminder


@router.delete("/reminders/{reminder_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete(reminder_id: uuid.UUID, db: DbSession, user: CurrentUser) -> None:
    """Delete a reminder and its alerts."""
    db.delete(_load(db, reminder_id, user))
    db.commit()


@router.post("/assignments", response_model=AssignmentOut, status_code=status.HTTP_201_CREATED)
def create_assignment(body: AssignmentArgs, db: DbSession, user: FacultyUser) -> AssignmentOut:
    """Post an assignment to a class. Faculty can only post to their own department."""
    if user.role == "faculty" and body.department != user.department:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not your department")
    try:
        task, created = post_assignment(db, user, body)
    except ReminderError as exc:
        raise _reject(exc) from exc
    db.commit()
    result = AssignmentOut.model_validate(task)
    result.reminders_created = created
    return result


@router.get("/assignments", response_model=AssignmentPage)
def list_assignments(
    db: DbSession,
    user: FacultyUser,
    mine: bool = False,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AssignmentPage:
    """List assignments that have not been deleted, soonest first."""
    filters: list[ColumnElement[bool]] = [Task.deleted_at.is_(None)]
    if mine:
        filters.append(Task.created_by == user.id)
    total = db.scalar(select(func.count()).select_from(Task).where(*filters)) or 0
    rows = db.scalars(
        select(Task).where(*filters).order_by(Task.due_at).limit(limit).offset(offset)
    ).all()
    return AssignmentPage(items=[AssignmentOut.model_validate(row) for row in rows], total=total)


@router.delete("/assignments/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_assignment(task_id: uuid.UUID, db: DbSession, user: FacultyUser) -> None:
    """Delete an assignment and cancel its reminders."""
    task = db.get(Task, task_id)
    if task is None or task.deleted_at is not None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Assignment not found")
    allowed = user.role == "admin" or task.created_by == user.id
    if not allowed and task.department != user.department:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not your department")
    cancel_assignment(db, task)
    db.commit()


@router.get("/notifications", response_model=NotificationPage)
def list_notifications(
    db: DbSession,
    user: CurrentUser,
    unread: bool = False,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> NotificationPage:
    """List delivered in-app notifications, newest first."""
    filters = [
        ReminderNotification.user_id == user.id,
        ReminderNotification.channel == "in_app",
        ReminderNotification.sent_at.is_not(None),
    ]
    if unread:
        filters.append(ReminderNotification.read_at.is_(None))
    total = db.scalar(select(func.count()).select_from(ReminderNotification).where(*filters)) or 0
    rows = db.scalars(
        select(ReminderNotification)
        .where(*filters)
        .order_by(ReminderNotification.sent_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return NotificationPage(
        items=[
            NotificationOut(
                id=row.id,
                reminder_id=row.reminder_id,
                title=row.payload["title"],
                due_at=row.payload["due_at"],
                sent_at=row.sent_at,  # type: ignore[arg-type]
                read_at=row.read_at,
            )
            for row in rows
        ],
        total=total,
    )


@router.post("/notifications/{notification_id}/read", status_code=status.HTTP_204_NO_CONTENT)
def mark_read(notification_id: uuid.UUID, db: DbSession, user: CurrentUser) -> None:
    """Mark a notification as read."""
    row = db.get(ReminderNotification, notification_id)
    if row is None or row.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Notification not found")
    row.read_at = row.read_at or utcnow()
    db.commit()
