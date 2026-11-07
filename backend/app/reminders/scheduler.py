"""Finding due notifications and delivering each one once.

Every tick does three things:

1. Create a notification row for each fire time that has arrived. The idempotency key is
   unique, so repeating this, from a duplicate tick or a second worker, adds nothing.
2. Move repeating reminders on to their next due date.
3. Claim unsent rows with FOR UPDATE SKIP LOCKED, deliver them, and record `sent_at`.

The key and the row lock guarantee one row and one claimant. They cannot make SMTP
transactional: a crash after the mail server accepts a message and before the commit
would send that email again on the next tick. Nothing is lost, and the copy carries the
same Message-ID, derived from the idempotency key, so mail clients that deduplicate show
it once. In-app delivery has no such gap, because the row itself is the notification.
"""

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from sqlalchemy import or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session, sessionmaker

from app.email import EmailSender
from app.logging import get_logger
from app.models import Reminder, ReminderNotification, User

log = get_logger(__name__)

CHANNELS = ("in_app", "email")
LOOKAHEAD = timedelta(days=30)
RECURRENCE_STEPS = {"daily": timedelta(days=1), "weekly": timedelta(days=7)}


class Publisher(Protocol):
    """Pushes a live message to a user's open sessions."""

    def publish(self, user_id: uuid.UUID, message: dict[str, Any]) -> None:
        """Send a message, if anyone is listening."""
        ...


class RedisPublisher:
    """Publishes to the Redis channel the WebSocket endpoint subscribes to."""

    def __init__(self, client: Any) -> None:
        """Keep the Redis client."""
        self._client = client

    def publish(self, user_id: uuid.UUID, message: dict[str, Any]) -> None:
        """Publish a message as JSON."""
        self._client.publish(f"notifications:{user_id}", json.dumps(message, default=str))


@dataclass(frozen=True)
class FireTime:
    """One moment a reminder should alert the user."""

    at: datetime
    marker: str


def fire_times(reminder: Reminder) -> Iterator[FireTime]:
    """List a reminder's alerts: each lead time, the due time, and any snooze.

    A snooze replaces the alerts before it. An alert that would have fired before the
    reminder even existed is skipped, so a reminder set for tomorrow does not immediately
    report that it is one week away.
    """
    moments = [
        FireTime(reminder.due_at - timedelta(minutes=minutes), f"before-{minutes}")
        for minutes in reminder.remind_before
    ]
    moments.append(FireTime(reminder.due_at, "due"))

    snoozed = reminder.snoozed_until
    for moment in moments:
        if moment.at < reminder.created_at or (snoozed and moment.at <= snoozed):
            continue
        yield moment
    if snoozed:
        yield FireTime(snoozed, f"snooze-{snoozed.isoformat()}")


def idempotency_key(reminder: Reminder, moment: FireTime, channel: str) -> str:
    """Identify one delivery. The due date is included so each recurrence gets its own."""
    return f"{reminder.id}:{reminder.due_at.isoformat()}:{moment.marker}:{channel}"


def _create_due_rows(session: Session, now: datetime) -> int:
    """Insert a row for every alert that has arrived, ignoring ones that already exist."""
    reminders = session.scalars(
        select(Reminder).where(
            Reminder.status == "pending",
            or_(Reminder.due_at <= now + LOOKAHEAD, Reminder.snoozed_until <= now),
        )
    ).all()
    created = 0
    for reminder in reminders:
        for moment in fire_times(reminder):
            if moment.at > now:
                continue
            for channel in CHANNELS:
                result = session.execute(
                    insert(ReminderNotification)
                    .values(
                        id=uuid.uuid4(),
                        reminder_id=reminder.id,
                        user_id=reminder.user_id,
                        channel=channel,
                        fire_at=moment.at,
                        idempotency_key=idempotency_key(reminder, moment, channel),
                        payload={
                            "title": reminder.title,
                            "subject": reminder.subject,
                            "due_at": reminder.due_at.isoformat(),
                        },
                    )
                    .on_conflict_do_nothing(index_elements=["idempotency_key"])
                    .returning(ReminderNotification.id)
                )
                created += 1 if result.first() else 0
    return created


def _advance_recurring(session: Session, now: datetime) -> None:
    """Move repeating reminders past their due date to the next occurrence."""
    overdue = session.scalars(
        select(Reminder).where(
            Reminder.status == "pending",
            Reminder.recurrence.is_not(None),
            Reminder.due_at <= now,
        )
    ).all()
    for reminder in overdue:
        step = RECURRENCE_STEPS[reminder.recurrence or "weekly"]
        while reminder.due_at <= now:
            reminder.due_at += step
        reminder.snoozed_until = None


def _format_body(payload: dict[str, Any], timezone: str) -> str:
    """Write the message, showing the due time in the user's current timezone."""
    due = datetime.fromisoformat(payload["due_at"]).astimezone(ZoneInfo(timezone))
    return f"{payload['title']} is due on {due.strftime('%a %d %b, %I:%M %p')} ({timezone})."


def _deliver(
    row: ReminderNotification, user: User, email: EmailSender, publisher: Publisher
) -> None:
    """Send one notification on its channel."""
    body = _format_body(row.payload, user.timezone)
    if row.channel == "email":
        email.send(
            user.email, f"Reminder: {row.payload['title']}", body, message_id=row.idempotency_key
        )
        return
    publisher.publish(
        user.id,
        {
            "type": "notification",
            "id": str(row.id),
            "title": row.payload["title"],
            "body": body,
            "due_at": row.payload["due_at"],
            "reminder_id": str(row.reminder_id),
        },
    )


def send_pending(
    session_factory: sessionmaker[Session],
    now: datetime,
    email: EmailSender,
    publisher: Publisher,
) -> int:
    """Deliver unsent notifications one at a time, each in its own transaction.

    SKIP LOCKED lets several workers share the queue without either waiting or doubling
    up. A failed send rolls back and is retried on the next tick, and it is skipped for
    the rest of this one, so one bad address cannot hold up everyone else's reminders.
    """
    sent = 0
    failed: set[uuid.UUID] = set()
    while True:
        with session_factory() as session:
            row = session.execute(
                select(ReminderNotification, User)
                .join(Reminder, ReminderNotification.reminder_id == Reminder.id)
                .join(User, ReminderNotification.user_id == User.id)
                .where(
                    ReminderNotification.sent_at.is_(None),
                    ReminderNotification.fire_at <= now,
                    Reminder.status == "pending",
                    User.is_active.is_(True),
                    ReminderNotification.id.not_in(failed),
                )
                .order_by(ReminderNotification.fire_at)
                .limit(1)
                .with_for_update(of=ReminderNotification, skip_locked=True)
            ).first()
            if row is None:
                return sent

            notification, user = row
            try:
                _deliver(notification, user, email, publisher)
            except Exception as exc:
                session.rollback()
                failed.add(notification.id)
                log.warning(
                    "notification_failed",
                    notification_id=str(notification.id),
                    channel=notification.channel,
                    error=str(exc),
                )
                continue
            notification.sent_at = now
            session.commit()
            sent += 1


def dispatch_due(
    session_factory: sessionmaker[Session],
    now: datetime,
    email: EmailSender,
    publisher: Publisher,
) -> int:
    """Run one scheduler tick. Returns the number of notifications sent."""
    with session_factory() as session:
        created = _create_due_rows(session, now)
        session.commit()
        _advance_recurring(session, now)
        session.commit()
    sent = send_pending(session_factory, now, email, publisher)
    if created or sent:
        log.info("reminders_dispatched", created=created, sent=sent)
    return sent
