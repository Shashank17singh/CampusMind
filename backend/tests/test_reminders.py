"""Tests for reminder parsing, scheduling, delivery, assignments and the WebSocket."""

import json
import smtplib
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from starlette.websockets import WebSocketDisconnect

from app.agents.state import HandlerInput, UserContext
from app.api.ws import get_notification_bus
from app.auth.tokens import create_access_token
from app.db import get_session_factory
from app.email import LoggingEmailSender, SmtpEmailSender
from app.llm import LLMError, get_llm
from app.main import create_app
from app.models import Reminder, ReminderNotification, Task, User
from app.reminders.parsing import (
    Clarification,
    ParsedReminder,
    extract_with_rules,
    parse_reminder,
    resolve_due,
)
from app.reminders.scheduler import (
    FireTime,
    RedisPublisher,
    _create_due_rows,
    dispatch_due,
    fire_times,
    idempotency_key,
    send_pending,
)
from app.reminders.service import (
    AssignmentArgs,
    ReminderAgent,
    ReminderArgs,
    ReminderError,
    cancel_assignment,
    create_reminder,
    list_reminders,
    mark_done,
    post_assignment,
    snooze,
)
from tests.conftest import auth_header
from tests.helpers import FakeLLM

IST = ZoneInfo("Asia/Kolkata")
# A Monday morning, so "Friday" and "next Monday" have clear answers.
NOW = datetime(2026, 11, 2, 10, 0, tzinfo=IST)


def due(text: str) -> datetime:
    """Resolve date words, failing the test if they need clarifying."""
    result = resolve_due(text, NOW)
    assert result.due_at is not None, result.clarification
    return result.due_at.astimezone(IST)


def at(day: int, hour: int = 23, minute: int = 59, month: int = 11) -> datetime:
    """Build a local datetime."""
    return datetime(2026, month, day, hour, minute, tzinfo=IST)


class TestDateResolution:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Friday 5pm", at(6, 17, 0)),
            ("on Friday at 5pm", at(6, 17, 0)),
            ("tomorrow at 9am", at(3, 9, 0)),
            ("next Monday", at(9)),
            ("next monday 10am", at(9, 10, 0)),
            ("this friday", at(6)),
            ("this friday 3pm", at(6, 15, 0)),
            ("in 2 hours", at(2, 12, 0)),
            ("15 november", at(15)),
            ("15th Nov at 6 pm", at(15, 18, 0)),
            ("Nov 15", at(15)),
            ("tonight", at(2, 21, 0)),
            ("tonight at 8", at(2, 20, 0)),
            ("friday evening", at(6, 18, 0)),
            ("tomorrow morning", at(3, 9, 0)),
        ],
    )
    def test_fifteen_phrasings(self, text: str, expected: datetime) -> None:
        assert due(text) == expected

    def test_a_day_with_no_time_means_the_end_of_that_day(self) -> None:
        assert due("by friday") == at(6)

    def test_tomorrow_alone_is_the_end_of_tomorrow(self) -> None:
        assert due("tomorrow") == at(3)

    def test_day_after_tomorrow(self) -> None:
        assert due("day after tomorrow") == at(4)

    def test_next_weekday_on_that_same_day_means_a_week_later(self) -> None:
        assert due("next monday") == at(9)

    def test_this_weekday_on_that_same_day_means_today(self) -> None:
        assert due("this monday 5pm") == at(2, 17, 0)

    def test_an_unambiguous_numeric_date_is_read_day_first(self) -> None:
        assert due("25/12/2026") == at(25, month=12)

    def test_the_result_is_timezone_aware(self) -> None:
        assert resolve_due("Friday 5pm", NOW).due_at.utcoffset() == timedelta(hours=5, minutes=30)

    def test_another_timezone_is_respected(self) -> None:
        result = resolve_due("Friday 5pm", NOW, "America/New_York")
        assert result.due_at.astimezone(ZoneInfo("America/New_York")).hour == 17

    @pytest.mark.parametrize(
        "text", ["03/04", "3/4/2027", "11-12", "soon", "sometime next month", "end of the week"]
    )
    def test_ambiguous_dates_ask_for_clarification(self, text: str) -> None:
        result = resolve_due(text, NOW)
        assert result.due_at is None
        assert result.clarification

    def test_a_past_date_is_refused(self) -> None:
        assert "already passed" in resolve_due("1 January 2020", NOW).clarification

    def test_no_date_asks_when(self) -> None:
        assert resolve_due(None, NOW).clarification == "When should I remind you?"

    def test_nonsense_asks_for_a_date(self) -> None:
        assert "couldn't read" in resolve_due("blorp", NOW).clarification


class TestRuleExtraction:
    @pytest.mark.parametrize(
        ("message", "title", "due_at"),
        [
            (
                "remind me to submit the DBMS assignment on Friday 5pm",
                "submit the DBMS assignment",
                at(6, 17, 0),
            ),
            ("Remind me to pay fees by tomorrow", "pay fees", at(3)),
            ("remind me about the viva next monday at 10am", "the viva", at(9, 10, 0)),
            ("set a reminder to call the dean in 2 hours", "call the dean", at(2, 12, 0)),
            ("please remind me to revise OS tonight", "revise OS", at(2, 21, 0)),
            ("reminder for lab record on 15 november", "lab record", at(15)),
            ("don't let me forget to return books on friday evening", "return books", at(6, 18, 0)),
        ],
    )
    def test_title_and_date_are_separated(self, message: str, title: str, due_at: datetime) -> None:
        fields = extract_with_rules(message)
        assert fields.title == title
        assert due(fields.due_text or "") == due_at

    def test_subject_code_is_found(self) -> None:
        assert extract_with_rules("submit DBMS assignment Friday 5pm").subject == "DBMS"

    def test_times_are_not_mistaken_for_subjects(self) -> None:
        assert extract_with_rules("call home at 5 PM tomorrow").subject is None

    @pytest.mark.parametrize(
        ("message", "recurrence"),
        [
            ("remind me about the lab record every friday at 5pm", "weekly"),
            ("weekly reminder for lab records on monday", "weekly"),
            ("remind me to drink water every day at 9am", "daily"),
            ("remind me once on friday", None),
        ],
    )
    def test_recurrence(self, message: str, recurrence: str | None) -> None:
        assert extract_with_rules(message).recurrence == recurrence


class TestParseReminder:
    llm_reply = json.dumps(
        {"title": "Submit DBMS assignment", "due_text": "Friday 5pm", "subject": "DBMS"}
    )

    def test_uses_the_llm_for_fields_and_dateparser_for_the_date(self) -> None:
        parsed = parse_reminder("submit DBMS Friday 5pm", FakeLLM([self.llm_reply]), NOW)
        assert isinstance(parsed, ParsedReminder)
        assert parsed.title == "Submit DBMS assignment"
        assert parsed.due_at.astimezone(IST) == at(6, 17, 0)
        assert parsed.subject == "DBMS"
        assert parsed.remind_before == (1440, 120)

    def test_date_words_are_removed_from_the_llm_title(self) -> None:
        reply = json.dumps(
            {"title": "Submit DBMS Assignment Tomorrow at 5pm", "due_text": "tomorrow at 5pm"}
        )
        message = "submit the assignment tomorrow at 5pm"
        parsed = parse_reminder(message, FakeLLM([reply]), NOW)
        assert isinstance(parsed, ParsedReminder)
        assert parsed.title == "Submit DBMS Assignment"

    def test_a_title_that_is_only_the_date_is_kept(self) -> None:
        reply = json.dumps({"title": "Friday 5pm", "due_text": "Friday 5pm"})
        parsed = parse_reminder("Friday 5pm", FakeLLM([reply]), NOW)
        assert isinstance(parsed, ParsedReminder)
        assert parsed.title == "Friday 5pm"

    def test_the_llm_never_supplies_the_date_itself(self) -> None:
        llm = FakeLLM([self.llm_reply])
        parse_reminder("submit it Friday 5pm", llm, NOW)
        assert "Do not work out the date yourself" in llm.calls[0]["system"]

    def test_llm_outage_falls_back_to_rules(self) -> None:
        parsed = parse_reminder("remind me to pay fees by tomorrow", FakeLLM([LLMError("x")]), NOW)
        assert isinstance(parsed, ParsedReminder)
        assert parsed.title == "pay fees"

    def test_invalid_llm_reply_falls_back_to_rules(self) -> None:
        parsed = parse_reminder("remind me to pay fees by tomorrow", FakeLLM(["nonsense"]), NOW)
        assert isinstance(parsed, ParsedReminder)

    def test_an_unreadable_llm_date_falls_back_to_the_rules_date(self) -> None:
        reply = json.dumps({"title": "Pay fees", "due_text": "the day of reckoning"})
        parsed = parse_reminder("pay fees by tomorrow", FakeLLM([reply]), NOW)
        assert isinstance(parsed, ParsedReminder)
        assert parsed.due_at.astimezone(IST) == at(3)

    def test_a_date_the_user_never_typed_is_ignored(self) -> None:
        reply = json.dumps({"title": "Pay the fee", "due_text": "next Friday at noon"})
        parsed = parse_reminder("pay the fee soon", FakeLLM([reply]), NOW)
        assert isinstance(parsed, Clarification)
        assert "soon" in parsed.question

    def test_an_invented_date_never_creates_a_reminder(self) -> None:
        reply = json.dumps({"title": "Pay the fee", "due_text": "never"})
        parsed = parse_reminder("pay the fee", FakeLLM([reply]), NOW)
        assert parsed == Clarification("When should I remind you?")

    def test_date_words_are_matched_ignoring_case(self) -> None:
        reply = json.dumps({"title": "Pay", "due_text": "friday 5pm"})
        assert isinstance(
            parse_reminder("Pay on Friday 5pm", FakeLLM([reply]), NOW), ParsedReminder
        )

    def test_ambiguous_date_asks_a_question(self) -> None:
        reply = json.dumps({"title": "Pay fees", "due_text": "03/04"})
        parsed = parse_reminder("pay fees on 03/04", FakeLLM([reply]), NOW)
        assert isinstance(parsed, Clarification)
        assert "03/04" in parsed.question

    def test_no_date_asks_a_question(self) -> None:
        reply = json.dumps({"title": "Pay fees", "due_text": None})
        assert isinstance(parse_reminder("pay fees", FakeLLM([reply]), NOW), Clarification)

    def test_past_date_asks_a_question(self) -> None:
        reply = json.dumps({"title": "Pay fees", "due_text": "1 January 2020"})
        result = parse_reminder("pay by 1 January 2020", FakeLLM([reply]), NOW)
        assert isinstance(result, Clarification)
        assert "already passed" in result.question


class SpyEmail:
    """Records what would have been sent."""

    def __init__(self, fail_for: set[str] | None = None) -> None:
        self.sent: list[tuple[str, str, str]] = []
        self.fail_for = fail_for or set()

    def send(self, to: str, subject: str, body: str) -> None:
        if to in self.fail_for:
            raise smtplib.SMTPException("rejected")
        self.sent.append((to, subject, body))


class SpyPublisher:
    """Records published messages."""

    def __init__(self) -> None:
        self.messages: list[tuple[uuid.UUID, dict[str, Any]]] = []

    def publish(self, user_id: uuid.UUID, message: dict[str, Any]) -> None:
        self.messages.append((user_id, message))


def make_reminder(db: Session, user: User, **fields: object) -> Reminder:
    """Store a reminder created at NOW."""
    defaults: dict[str, object] = {
        "user_id": user.id,
        "title": "Submit DBMS",
        "due_at": NOW + timedelta(days=3),
        "created_at": NOW,
        "remind_before": [1440, 120],
    }
    reminder = Reminder(**{**defaults, **fields})
    db.add(reminder)
    db.commit()
    return reminder


def tick(when: datetime, email: SpyEmail | None = None, publisher: SpyPublisher | None = None):
    """Run one scheduler tick."""
    spy_email = email or SpyEmail()
    spy_publisher = publisher or SpyPublisher()
    sent = dispatch_due(get_session_factory(), when, spy_email, spy_publisher)
    return sent, spy_email, spy_publisher


class TestFireTimes:
    def reminder(self, **fields: object) -> Reminder:
        defaults: dict[str, object] = {
            "id": uuid.uuid4(),
            "due_at": NOW + timedelta(days=3),
            "created_at": NOW,
            "remind_before": [1440, 120],
            "snoozed_until": None,
        }
        return Reminder(**{**defaults, **fields})

    def test_lead_times_and_the_due_time(self) -> None:
        markers = [moment.marker for moment in fire_times(self.reminder())]
        assert markers == ["before-1440", "before-120", "due"]

    def test_alerts_from_before_the_reminder_existed_are_skipped(self) -> None:
        reminder = self.reminder(due_at=NOW + timedelta(hours=5))
        assert [moment.marker for moment in fire_times(reminder)] == ["before-120", "due"]

    def test_snooze_replaces_earlier_alerts_and_keeps_the_due_alert(self) -> None:
        snoozed = NOW + timedelta(days=2, hours=23)
        markers = [m.marker for m in fire_times(self.reminder(snoozed_until=snoozed))]
        assert markers == ["due", f"snooze-{snoozed.isoformat()}"]

    def test_each_recurrence_gets_a_different_key(self) -> None:
        moment = FireTime(NOW, "due")
        first = self.reminder()
        second = self.reminder(id=first.id, due_at=first.due_at + timedelta(days=7))
        assert idempotency_key(first, moment, "email") != idempotency_key(second, moment, "email")


class TestDelivery:
    @pytest.fixture
    def user(self, make_user: Callable[..., User]) -> User:
        return make_user(email="asha@college.edu")

    def test_nothing_fires_before_its_time(self, db: Session, user: User) -> None:
        make_reminder(db, user)
        assert tick(NOW + timedelta(hours=1))[0] == 0

    def test_a_due_alert_goes_to_both_channels(self, db: Session, user: User) -> None:
        make_reminder(db, user)
        sent, email, publisher = tick(NOW + timedelta(days=2))
        assert sent == 2
        assert len(email.sent) == 1
        assert email.sent[0][0] == "asha@college.edu"
        assert publisher.messages[0][1]["type"] == "notification"

    def test_repeat_ticks_send_nothing_new(self, db: Session, user: User) -> None:
        make_reminder(db, user)
        when = NOW + timedelta(days=2)
        tick(when)
        sent, email, publisher = tick(when)
        assert sent == 0
        assert email.sent == []
        assert publisher.messages == []

    def test_a_later_tick_sends_only_the_new_alert(self, db: Session, user: User) -> None:
        make_reminder(db, user)
        tick(NOW + timedelta(days=2))
        sent, _, _ = tick(NOW + timedelta(days=2, hours=23))
        assert sent == 2

    def test_a_tick_after_downtime_sends_each_missed_alert_once(
        self, db: Session, user: User
    ) -> None:
        make_reminder(db, user)
        assert tick(NOW + timedelta(days=4))[0] == 6
        assert tick(NOW + timedelta(days=4))[0] == 0

    def test_two_workers_cannot_claim_the_same_notification(self, db: Session, user: User) -> None:
        make_reminder(db, user)
        when = NOW + timedelta(days=2)
        factory = get_session_factory()
        with factory() as session:
            _create_due_rows(session, when)
            session.commit()

        email = SpyEmail()
        with factory() as other_worker:
            other_worker.scalars(
                select(ReminderNotification)
                .where(ReminderNotification.channel == "email")
                .with_for_update()
            ).one()
            # The email row is locked, so this worker skips it instead of waiting or doubling up.
            assert send_pending(factory, when, email, SpyPublisher()) == 1
            assert email.sent == []
            other_worker.rollback()

        assert send_pending(factory, when, email, SpyPublisher()) == 1
        assert len(email.sent) == 1

    def test_a_failed_email_is_retried_on_the_next_tick(self, db: Session, user: User) -> None:
        make_reminder(db, user)
        when = NOW + timedelta(days=2)
        broken = SpyEmail(fail_for={user.email})
        assert tick(when, broken)[0] == 1
        working = SpyEmail()
        assert tick(when, working)[0] == 1
        assert len(working.sent) == 1

    def test_one_bad_address_does_not_block_others(
        self, db: Session, user: User, make_user: Callable[..., User]
    ) -> None:
        other = make_user(email="ravi@college.edu")
        make_reminder(db, user)
        make_reminder(db, other)
        email = SpyEmail(fail_for={user.email})
        tick(NOW + timedelta(days=2), email)
        assert [item[0] for item in email.sent] == ["ravi@college.edu"]

    def test_done_reminders_stop_alerting(self, db: Session, user: User) -> None:
        make_reminder(db, user, status="done")
        assert tick(NOW + timedelta(days=2))[0] == 0

    def test_a_reminder_finished_after_its_row_exists_is_not_sent(
        self, db: Session, user: User
    ) -> None:
        reminder = make_reminder(db, user)
        when = NOW + timedelta(days=2)
        email = SpyEmail(fail_for={user.email})
        tick(when, email)
        reminder.status = "done"
        db.commit()
        assert tick(when)[0] == 0

    def test_inactive_users_get_nothing(self, db: Session, make_user: Callable[..., User]) -> None:
        make_reminder(db, make_user(is_active=False))
        assert tick(NOW + timedelta(days=2))[0] == 0

    def test_times_use_the_users_current_timezone(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        make_reminder(db, user, due_at=at(5, 12, 0))
        user.timezone = "America/New_York"
        db.commit()
        _, email, _ = tick(at(4, 13, 0))
        assert "01:30 AM" in email.sent[0][2]
        assert "America/New_York" in email.sent[0][2]

    def test_changing_timezone_does_not_move_the_deadline(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        reminder = make_reminder(db, user)
        before = reminder.due_at
        user.timezone = "Europe/London"
        db.commit()
        db.expire_all()
        assert db.get_one(Reminder, reminder.id).due_at == before

    def test_snoozed_alert_fires_at_the_new_time(self, db: Session, user: User) -> None:
        reminder = make_reminder(db, user, due_at=NOW + timedelta(hours=5))
        snooze(reminder, 30, now=NOW + timedelta(hours=1))
        db.commit()
        assert tick(NOW + timedelta(hours=1, minutes=10))[0] == 0
        assert tick(NOW + timedelta(hours=1, minutes=31))[0] == 2

    def test_redis_publisher_targets_the_users_channel(self) -> None:
        import fakeredis

        client = fakeredis.FakeRedis(decode_responses=True)
        pubsub = client.pubsub()
        user_id = uuid.uuid4()
        pubsub.subscribe(f"notifications:{user_id}")
        pubsub.get_message(timeout=1)
        RedisPublisher(client).publish(user_id, {"type": "notification"})
        assert json.loads(pubsub.get_message(timeout=1)["data"]) == {"type": "notification"}


class TestRecurrence:
    def test_each_week_gets_its_own_alerts(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        make_reminder(
            db, user, due_at=NOW + timedelta(days=1), remind_before=[], recurrence="weekly"
        )
        assert tick(NOW + timedelta(days=1, minutes=1))[0] == 2
        assert tick(NOW + timedelta(days=1, minutes=2))[0] == 0
        assert tick(NOW + timedelta(days=8, minutes=1))[0] == 2

    def test_due_date_moves_forward_after_it_passes(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        reminder = make_reminder(
            db, make_user(), due_at=NOW + timedelta(days=1), recurrence="weekly"
        )
        tick(NOW + timedelta(days=1, minutes=1))
        db.expire_all()
        assert db.get_one(Reminder, reminder.id).due_at == NOW + timedelta(days=8)

    def test_downtime_skips_to_the_next_future_occurrence(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        reminder = make_reminder(
            db, make_user(), due_at=NOW + timedelta(days=1), recurrence="weekly"
        )
        tick(NOW + timedelta(days=30))
        db.expire_all()
        assert db.get_one(Reminder, reminder.id).due_at > NOW + timedelta(days=30)

    def test_daily_reminders_repeat_daily(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        reminder = make_reminder(
            db, make_user(), due_at=NOW + timedelta(hours=1), recurrence="daily"
        )
        tick(NOW + timedelta(hours=2))
        db.expire_all()
        assert db.get_one(Reminder, reminder.id).due_at == NOW + timedelta(days=1, hours=1)

    def test_marking_a_repeating_reminder_done_moves_it_on(self) -> None:
        reminder = Reminder(due_at=NOW, recurrence="weekly", status="pending")
        mark_done(reminder, now=NOW)
        assert reminder.status == "pending"
        assert reminder.due_at == NOW + timedelta(days=7)

    def test_marking_a_one_off_done_finishes_it(self) -> None:
        reminder = Reminder(due_at=NOW, recurrence=None, status="pending")
        mark_done(reminder)
        assert reminder.status == "done"


class TestService:
    def args(self, **fields: object) -> ReminderArgs:
        return ReminderArgs.model_validate(
            {"title": "x", "due_at": NOW + timedelta(days=1), **fields}
        )

    def test_past_dates_are_refused(self, db: Session, make_user: Callable[..., User]) -> None:
        with pytest.raises(ReminderError, match="already passed"):
            create_reminder(db, make_user().id, self.args(due_at=NOW - timedelta(days=1)), now=NOW)

    def test_a_date_without_a_timezone_is_refused(self) -> None:
        with pytest.raises(ValueError, match="timezone"):
            self.args(due_at="2026-12-01T10:00:00")

    @pytest.mark.parametrize("lead", [[0], [-5], [50000], [1, 2, 3, 4, 5, 6]])
    def test_bad_lead_times_are_refused(self, lead: list[int]) -> None:
        with pytest.raises(ValueError, match="lead times"):
            self.args(remind_before=lead)

    def test_views_separate_upcoming_overdue_and_done(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        make_reminder(db, user, title="future", due_at=NOW + timedelta(days=1))
        make_reminder(db, user, title="late", due_at=NOW - timedelta(days=1))
        make_reminder(db, user, title="finished", due_at=NOW - timedelta(days=2), status="done")
        titles = {
            view: [r.title for r in list_reminders(db, user.id, view, now=NOW)[0]]
            for view in ("upcoming", "overdue", "done")
        }
        assert titles == {"upcoming": ["future"], "overdue": ["late"], "done": ["finished"]}

    def test_a_finished_reminder_cannot_be_snoozed(self) -> None:
        with pytest.raises(ReminderError, match="pending"):
            snooze(Reminder(status="done"), 10)


class TestAssignments:
    def args(self, **fields: object) -> AssignmentArgs:
        return AssignmentArgs.model_validate(
            {
                "title": "Lab 3",
                "department": "CSE",
                "semester": 3,
                "due_at": NOW + timedelta(days=5),
                **fields,
            }
        )

    def test_every_student_in_the_class_gets_a_reminder(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        faculty = make_user("faculty", department="CSE")
        in_class = [make_user(department="CSE", semester=3) for _ in range(3)]
        make_user(department="CSE", semester=5)
        make_user(department="ECE", semester=3)
        make_user("faculty", department="CSE", semester=3)
        make_user(department="CSE", semester=3, is_active=False)

        task, created = post_assignment(db, faculty, self.args(), now=NOW)
        db.commit()

        assert created == 3
        rows = db.scalars(select(Reminder).where(Reminder.task_id == task.id)).all()
        assert {row.user_id for row in rows} == {student.id for student in in_class}

    def test_a_section_narrows_the_class(self, db: Session, make_user: Callable[..., User]) -> None:
        faculty = make_user("faculty", department="CSE")
        make_user(department="CSE", semester=3, section="A")
        make_user(department="CSE", semester=3, section="B")
        _, created = post_assignment(db, faculty, self.args(section="A"), now=NOW)
        assert created == 1

    def test_an_empty_class_is_not_an_error(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        faculty = make_user("faculty", department="CSE")
        assert post_assignment(db, faculty, self.args(), now=NOW)[1] == 0

    def test_a_past_deadline_is_refused(self, db: Session, make_user: Callable[..., User]) -> None:
        with pytest.raises(ReminderError):
            post_assignment(
                db,
                make_user("faculty"),
                self.args(due_at=NOW - timedelta(days=1)),
                now=NOW,
            )

    def test_deleting_an_assignment_stops_unsent_alerts(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        faculty = make_user("faculty", department="CSE")
        make_user(department="CSE", semester=3)
        task, _ = post_assignment(db, faculty, self.args(), now=NOW)
        db.commit()

        when = NOW + timedelta(days=4, hours=1)
        student = db.scalars(select(User).where(User.role == "student")).one()
        tick(when, SpyEmail(fail_for={student.email}))

        cancel_assignment(db, task)
        db.commit()
        assert tick(when)[0] == 0
        assert db.scalar(select(func.count()).select_from(Reminder)) == 1
        reminder = db.scalars(select(Reminder)).one()
        assert reminder.status == "cancelled"


class TestAgent:
    @pytest.fixture
    def agent_user(self, make_user: Callable[..., User]) -> UserContext:
        return UserContext(id=make_user().id)

    def request(self, user: UserContext, text: str) -> HandlerInput:
        return HandlerInput(text, (), user, (), lambda _: None)

    def test_creates_a_reminder(self, db: Session, agent_user: UserContext) -> None:
        reply = json.dumps({"title": "Pay fees", "due_text": "tomorrow"})
        agent = ReminderAgent(FakeLLM([reply]), get_session_factory())
        result = agent.handle(self.request(agent_user, "remind me to pay fees tomorrow"))
        assert result.text.startswith("Done.")
        stored = db.scalar(select(Reminder).where(Reminder.user_id == agent_user.id))
        assert stored is not None
        assert stored.source_text == "remind me to pay fees tomorrow"

    def test_passes_on_a_clarifying_question(self, agent_user: UserContext) -> None:
        reply = json.dumps({"title": "Pay fees", "due_text": "soon"})
        agent = ReminderAgent(FakeLLM([reply]), get_session_factory())
        assert "vague" in agent.handle(self.request(agent_user, "pay fees soon")).text

    def test_lists_upcoming_reminders(self, db: Session, agent_user: UserContext) -> None:
        user = db.get_one(User, agent_user.id)
        make_reminder(db, user, title="Viva", due_at=datetime.now(UTC) + timedelta(days=1))
        agent = ReminderAgent(FakeLLM(), get_session_factory())
        result = agent.handle(self.request(agent_user, "what are my deadlines?"))
        assert "Viva" in result.text

    def test_an_empty_list_is_said_plainly(self, agent_user: UserContext) -> None:
        agent = ReminderAgent(FakeLLM(), get_session_factory())
        text = agent.handle(self.request(agent_user, "show my reminders")).text
        assert text == "You have no upcoming reminders."


class TestApi:
    @pytest.fixture
    def api(self, db: Session) -> TestClient:
        app = create_app()
        replies = [
            json.dumps({"title": "Submit DBMS", "due_text": "tomorrow 5pm"}),
            json.dumps({"title": "Submit DBMS", "due_text": "soon"}),
        ]
        app.dependency_overrides[get_llm] = lambda: FakeLLM(list(replies))
        return TestClient(app)

    def future(self) -> str:
        return (datetime.now(UTC) + timedelta(days=2)).isoformat()

    def create(self, api: TestClient, user: User) -> dict:
        response = api.post(
            "/api/v1/reminders",
            headers=auth_header(user),
            json={"title": "Viva", "due_at": self.future()},
        )
        assert response.status_code == 201
        return response.json()

    def test_create_and_list(self, api: TestClient, make_user: Callable[..., User]) -> None:
        user = make_user()
        reminder = self.create(api, user)
        listed = api.get("/api/v1/reminders", headers=auth_header(user)).json()
        assert [item["id"] for item in listed["items"]] == [reminder["id"]]

    def test_past_date_is_422(self, api: TestClient, make_user: Callable[..., User]) -> None:
        response = api.post(
            "/api/v1/reminders",
            headers=auth_header(make_user()),
            json={"title": "x", "due_at": "2020-01-01T00:00:00+00:00"},
        )
        assert response.status_code == 422

    def test_parse_creates_a_reminder(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = api.post(
            "/api/v1/reminders/parse",
            headers=auth_header(make_user()),
            json={"text": "remind me to submit DBMS tomorrow at 5pm"},
        )
        assert response.status_code == 201
        assert response.json()["reminder"]["title"] == "Submit DBMS"

    def test_parse_can_return_a_question(self, db: Session, make_user: Callable[..., User]) -> None:
        app = create_app()
        reply = json.dumps({"title": "x", "due_text": "soon"})
        app.dependency_overrides[get_llm] = lambda: FakeLLM([reply])
        response = TestClient(app).post(
            "/api/v1/reminders/parse",
            headers=auth_header(make_user()),
            json={"text": "do it soon"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["reminder"] is None
        assert "vague" in body["clarification"]

    def test_done_moves_it_to_the_done_tab(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        reminder = self.create(api, user)
        headers = auth_header(user)
        api.post(f"/api/v1/reminders/{reminder['id']}/done", headers=headers)
        done = api.get("/api/v1/reminders?view=done", headers=headers).json()
        assert done["total"] == 1

    def test_snooze(self, api: TestClient, make_user: Callable[..., User]) -> None:
        user = make_user()
        reminder = self.create(api, user)
        response = api.post(
            f"/api/v1/reminders/{reminder['id']}/snooze",
            headers=auth_header(user),
            json={"minutes": 30},
        )
        assert response.json()["snoozed_until"] is not None

    def test_delete(self, api: TestClient, make_user: Callable[..., User]) -> None:
        user = make_user()
        reminder = self.create(api, user)
        headers = auth_header(user)
        assert api.delete(f"/api/v1/reminders/{reminder['id']}", headers=headers).status_code == 204
        assert api.get("/api/v1/reminders", headers=headers).json()["total"] == 0

    @pytest.mark.parametrize("action", ["done", "snooze"])
    def test_other_users_cannot_touch_a_reminder(
        self, api: TestClient, make_user: Callable[..., User], action: str
    ) -> None:
        reminder = self.create(api, make_user())
        response = api.post(
            f"/api/v1/reminders/{reminder['id']}/{action}",
            headers=auth_header(make_user()),
            json={"minutes": 5},
        )
        assert response.status_code == 404

    def test_faculty_can_post_to_their_department(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        make_user(department="CSE", semester=3)
        response = api.post(
            "/api/v1/assignments",
            headers=auth_header(make_user("faculty", department="CSE")),
            json={"title": "Lab", "department": "CSE", "semester": 3, "due_at": self.future()},
        )
        assert response.status_code == 201
        assert response.json()["reminders_created"] == 1

    def test_faculty_cannot_post_to_another_department(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = api.post(
            "/api/v1/assignments",
            headers=auth_header(make_user("faculty", department="ECE")),
            json={"title": "Lab", "department": "CSE", "semester": 3, "due_at": self.future()},
        )
        assert response.status_code == 403

    def test_students_cannot_post_assignments(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = api.post(
            "/api/v1/assignments",
            headers=auth_header(make_user(department="CSE")),
            json={"title": "Lab", "department": "CSE", "semester": 3, "due_at": self.future()},
        )
        assert response.status_code == 403

    def test_deleted_assignments_leave_the_list(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        faculty = make_user("faculty", department="CSE")
        headers = auth_header(faculty)
        created = api.post(
            "/api/v1/assignments",
            headers=headers,
            json={"title": "Lab", "department": "CSE", "semester": 3, "due_at": self.future()},
        ).json()
        assert (
            api.delete(f"/api/v1/assignments/{created['id']}", headers=headers).status_code == 204
        )
        assert api.get("/api/v1/assignments", headers=headers).json()["total"] == 0
        assert db.get_one(Task, uuid.UUID(created["id"])).deleted_at is not None

    def test_faculty_cannot_delete_another_departments_assignment(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        created = api.post(
            "/api/v1/assignments",
            headers=auth_header(make_user("faculty", department="CSE")),
            json={"title": "Lab", "department": "CSE", "semester": 3, "due_at": self.future()},
        ).json()
        response = api.delete(
            f"/api/v1/assignments/{created['id']}",
            headers=auth_header(make_user("faculty", department="ECE")),
        )
        assert response.status_code == 403

    def test_notifications_list_and_read(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        make_reminder(db, user)
        tick(NOW + timedelta(days=2))
        headers = auth_header(user)
        listed = api.get("/api/v1/notifications?unread=true", headers=headers).json()
        assert listed["total"] == 1
        notification = listed["items"][0]
        assert notification["title"] == "Submit DBMS"
        read = api.post(f"/api/v1/notifications/{notification['id']}/read", headers=headers)
        assert read.status_code == 204
        assert api.get("/api/v1/notifications?unread=true", headers=headers).json()["total"] == 0

    def test_other_users_cannot_read_a_notification(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        make_reminder(db, user)
        tick(NOW + timedelta(days=2))
        notification = api.get("/api/v1/notifications", headers=auth_header(user)).json()["items"][
            0
        ]
        response = api.post(
            f"/api/v1/notifications/{notification['id']}/read",
            headers=auth_header(make_user()),
        )
        assert response.status_code == 404


class FakeBus:
    """Yields fixed messages, then stays open like a real subscription."""

    def __init__(self, messages: list[dict[str, Any]]) -> None:
        self.messages = messages
        self.subscribed: list[uuid.UUID] = []

    async def subscribe(self, user_id: uuid.UUID) -> AsyncIterator[dict[str, Any]]:
        import asyncio

        self.subscribed.append(user_id)
        for message in self.messages:
            yield message
        await asyncio.Event().wait()


class TestWebSocket:
    def client(self, bus: FakeBus) -> TestClient:
        app = create_app()
        app.dependency_overrides[get_notification_bus] = lambda: bus
        return TestClient(app)

    def test_a_valid_token_receives_notifications(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        bus = FakeBus([{"type": "notification", "title": "Viva"}])
        token = create_access_token(user.id, user.role)
        with self.client(bus).websocket_connect("/api/v1/ws/notifications") as socket:
            socket.send_json({"type": "auth", "token": token})
            assert socket.receive_json() == {"type": "ready"}
            assert socket.receive_json()["title"] == "Viva"
        assert bus.subscribed == [user.id]

    def test_ping_gets_a_pong(self, db: Session, make_user: Callable[..., User]) -> None:
        user = make_user()
        token = create_access_token(user.id, user.role)
        with self.client(FakeBus([])).websocket_connect("/api/v1/ws/notifications") as socket:
            socket.send_json({"type": "auth", "token": token})
            socket.receive_json()
            socket.send_json({"type": "ping"})
            assert socket.receive_json() == {"type": "pong"}

    @pytest.mark.parametrize(
        "first",
        [
            {"type": "auth", "token": "not-a-token"},
            {"type": "auth"},
            {"type": "ping"},
            "just a string",
        ],
    )
    def test_bad_first_messages_are_closed_with_4401(self, first: object) -> None:
        with (
            self.client(FakeBus([])).websocket_connect("/api/v1/ws/notifications") as socket,
            pytest.raises(WebSocketDisconnect) as closed,
        ):
            socket.send_json(first)
            socket.receive_json()
        assert closed.value.code == 4401

    def test_an_inactive_user_is_refused(self, db: Session, make_user: Callable[..., User]) -> None:
        user = make_user(is_active=False)
        token = create_access_token(user.id, user.role)
        with (
            self.client(FakeBus([])).websocket_connect("/api/v1/ws/notifications") as socket,
            pytest.raises(WebSocketDisconnect) as closed,
        ):
            socket.send_json({"type": "auth", "token": token})
            socket.receive_json()
        assert closed.value.code == 4401


class TestEmail:
    def test_smtp_sender_uses_starttls_and_login(self, monkeypatch: pytest.MonkeyPatch) -> None:
        events: list[str] = []

        class FakeSmtp:
            def __init__(self, host: str, port: int, timeout: int) -> None:
                events.append(f"connect {host}:{port}")

            def __enter__(self) -> "FakeSmtp":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def starttls(self) -> None:
                events.append("starttls")

            def login(self, username: str, password: str) -> None:
                events.append(f"login {username}")

            def send_message(self, message: Any) -> None:
                events.append(f"send {message['To']}")

        monkeypatch.setattr(smtplib, "SMTP", FakeSmtp)
        SmtpEmailSender("mail.test", 587, "user", "pw", "from@test").send("a@b.c", "Hi", "Body")
        assert events == ["connect mail.test:587", "starttls", "login user", "send a@b.c"]

    def test_logging_sender_sends_nothing_and_does_not_fail(self) -> None:
        LoggingEmailSender().send("a@b.c", "Hi", "Body")
