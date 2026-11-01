"""Tests for the study schedule, re-planning, export, the agent and the API."""

import json
import random
import uuid
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, date, datetime, time, timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.planner import (
    PlanInputs,
    PlannerAgent,
    PlannerError,
    SyllabusOutline,
    collect_topics,
    create_plan,
    export_ics,
    find_syllabus,
    load_outline,
    replan,
)
from app.agents.scheduler import (
    GENERAL_TOPIC,
    LEARN_MINUTES,
    PRACTICE_TOPIC,
    Busy,
    Draft,
    Exam,
    Topic,
    build_schedule,
    to_ics,
)
from app.agents.state import HandlerInput, UserContext
from app.db import get_session_factory
from app.llm import LLMError, get_llm
from app.main import create_app
from app.models import PlanItem, StudyPlan, User
from tests.conftest import auth_header
from tests.helpers import FakeLLM
from tests.test_vectorstore import add_ready_document

START = date(2026, 11, 2)  # A Monday.
HOURS = dict.fromkeys(range(7), 4.0)

OUTLINE = {
    "subjects": [
        {
            "name": "DBMS",
            "units": [
                {"name": "Unit 1", "topics": ["ER model", "Relational algebra"]},
                {"name": "Unit 2", "topics": ["Normalization", "Transactions"]},
            ],
        },
        {"name": "OS", "units": [{"name": "Unit 1", "topics": ["Scheduling", "Paging"]}]},
    ]
}


def minutes_of(item: Draft | PlanItem) -> int:
    """Return an item's length."""
    start = item.start if isinstance(item, Draft) else item.start_time
    end = item.end if isinstance(item, Draft) else item.end_time
    return (datetime.combine(date.min, end) - datetime.combine(date.min, start)).seconds // 60


def total_by_day(items: list[Draft]) -> dict[date, int]:
    """Sum the minutes scheduled on each day."""
    totals: dict[date, int] = defaultdict(int)
    for item in items:
        totals[item.date] += minutes_of(item)
    return totals


def topics_for(subject: str, count: int, difficulty: int = 3) -> list[Topic]:
    """Build a list of topics."""
    return [Topic(subject, f"{subject} topic {n}", difficulty) for n in range(count)]


class TestSchedulerRules:
    def schedule(self, **overrides: object):
        fields: dict[str, object] = {
            "topics": topics_for("DBMS", 4) + topics_for("OS", 3, difficulty=5),
            "exams": [
                Exam("DBMS", START + timedelta(days=14)),
                Exam("OS", START + timedelta(days=18)),
            ],
            "hours_by_weekday": HOURS,
            "start": START,
            **overrides,
        }
        return build_schedule(**fields)  # type: ignore[arg-type]

    def test_plan_fits_in_available_hours(self) -> None:
        result = self.schedule()
        assert all(minutes <= 240 for minutes in total_by_day(result.items).values())

    def test_a_day_with_no_hours_gets_nothing(self) -> None:
        hours = {**HOURS, 6: 0.0}
        result = self.schedule(hours_by_weekday=hours)
        assert all(item.date.weekday() != 6 for item in result.items)

    def test_uneven_hours_are_respected_per_weekday(self) -> None:
        hours = {0: 1.0, 1: 2.0, 2: 3.0, 3: 1.5, 4: 0.5, 5: 6.0, 6: 4.0}
        for item_day, minutes in total_by_day(self.schedule(hours_by_weekday=hours).items).items():
            assert minutes <= hours[item_day.weekday()] * 60

    def test_last_two_days_before_each_exam_are_revision_only(self) -> None:
        result = self.schedule()
        exams = [START + timedelta(days=14), START + timedelta(days=18)]
        blocked = {exam - timedelta(days=n) for exam in exams for n in (1, 2)}
        learned_days = {item.date for item in result.items if item.type == "learn"}
        assert not learned_days & blocked
        assert {item.type for item in result.items if item.date in blocked} <= {
            "revise",
            "practice",
        }

    def test_revision_days_are_actually_used(self) -> None:
        result = self.schedule()
        exam = START + timedelta(days=14)
        assert total_by_day(result.items)[exam - timedelta(days=1)] == 240

    def test_nothing_is_scheduled_on_an_exam_day(self) -> None:
        exams = {START + timedelta(days=14), START + timedelta(days=18)}
        assert not {item.date for item in self.schedule().items} & exams

    def test_each_subject_is_learned_before_its_revision_window(self) -> None:
        exams = {"DBMS": START + timedelta(days=14), "OS": START + timedelta(days=18)}
        for item in self.schedule().items:
            if item.type == "learn":
                assert item.date < exams[item.subject] - timedelta(days=2)

    def test_every_topic_gets_its_full_learning_time(self) -> None:
        result = self.schedule()
        learned: dict[str, int] = defaultdict(int)
        for item in result.items:
            if item.type == "learn":
                learned[item.topic] += minutes_of(item)
        assert learned["DBMS topic 0"] == LEARN_MINUTES[3]
        assert learned["OS topic 0"] == LEARN_MINUTES[5]

    def test_spaced_repetition_follows_one_three_seven_days(self) -> None:
        topic = Topic("DBMS", "Normalization", 2)
        result = build_schedule([topic], [Exam("DBMS", START + timedelta(days=20))], HOURS, START)
        learned = next(item.date for item in result.items if item.type == "learn")
        revisions = sorted(
            item.date
            for item in result.items
            if item.type == "revise" and item.date < START + timedelta(days=18)
        )
        assert revisions == [learned + timedelta(days=n) for n in (1, 3, 7)]

    def test_revisions_never_land_after_the_exam(self) -> None:
        exam = START + timedelta(days=5)
        result = build_schedule([Topic("DBMS", "ER model", 2)], [Exam("DBMS", exam)], HOURS, START)
        assert all(item.date < exam for item in result.items)

    def test_practice_is_scheduled_once_per_subject(self) -> None:
        practice = [i for i in self.schedule().items if i.type == "practice"]
        assert sorted(item.subject for item in practice) == ["DBMS", "OS"]
        assert all(item.topic == PRACTICE_TOPIC for item in practice)

    def test_clock_times_never_overlap(self) -> None:
        by_day: dict[date, list[Draft]] = defaultdict(list)
        for item in self.schedule().items:
            by_day[item.date].append(item)
        for items in by_day.values():
            ordered = sorted(items, key=lambda item: item.start)
            for first, second in zip(ordered, ordered[1:], strict=False):
                assert first.end <= second.start

    def test_the_same_input_gives_the_same_plan(self) -> None:
        assert self.schedule().items == self.schedule().items

    def test_harder_topics_with_closer_exams_come_first(self) -> None:
        topics = [Topic("Easy", "easy one", 1), Topic("Hard", "hard one", 5)]
        exams = [Exam("Easy", START + timedelta(days=20)), Exam("Hard", START + timedelta(days=8))]
        result = build_schedule(topics, exams, dict.fromkeys(range(7), 1.0), START)
        learn = [item for item in result.items if item.type == "learn"]
        assert learn[0].topic == "hard one"

    def test_a_big_topic_splits_across_days(self) -> None:
        topic = Topic("DBMS", "Long topic", 5)
        result = build_schedule(
            [topic], [Exam("DBMS", START + timedelta(days=15))], dict.fromkeys(range(7), 1.0), START
        )
        days = [item.date for item in result.items if item.type == "learn"]
        assert len(set(days)) > 1

    def test_topics_that_cannot_fit_are_reported_not_dropped_silently(self) -> None:
        topics = topics_for("DBMS", 30, difficulty=5)
        result = build_schedule(
            topics, [Exam("DBMS", START + timedelta(days=6))], dict.fromkeys(range(7), 1.0), START
        )
        assert result.unscheduled
        learned = {item.topic for item in result.items if item.type == "learn"}
        assert not learned & {topic.name for topic in result.unscheduled}

    def test_a_subject_with_no_topics_still_gets_revision_days(self) -> None:
        result = build_schedule([], [Exam("Maths", START + timedelta(days=5))], HOURS, START)
        assert {item.topic for item in result.items} == {PRACTICE_TOPIC, GENERAL_TOPIC}

    def test_past_exams_are_ignored(self) -> None:
        result = build_schedule(
            topics_for("DBMS", 2), [Exam("DBMS", START - timedelta(days=1))], HOURS, START
        )
        assert result.items == []
        assert len(result.unscheduled) == 2

    def test_busy_time_reduces_a_day_and_shifts_its_start(self) -> None:
        result = build_schedule(
            [Topic("DBMS", "ER", 2)],
            [Exam("DBMS", START + timedelta(days=10))],
            dict.fromkeys(range(7), 2.0),
            START,
            busy={START: Busy(90, time(10, 30))},
        )
        first_day = [item for item in result.items if item.date == START]
        assert all(item.start >= time(10, 30) for item in first_day)
        assert sum(minutes_of(item) for item in first_day) <= 30

    def test_exam_tomorrow_only_leaves_a_revision_day(self) -> None:
        result = build_schedule(
            [Topic("DBMS", "ER", 2)], [Exam("DBMS", START + timedelta(days=1))], HOURS, START
        )
        assert result.unscheduled == [Topic("DBMS", "ER", 2)]
        assert {item.type for item in result.items} <= {"revise", "practice"}

    @pytest.mark.parametrize("seed", range(25))
    def test_rules_hold_on_random_inputs(self, seed: int) -> None:
        rng = random.Random(seed)
        subjects = [f"S{n}" for n in range(rng.randint(1, 5))]
        exams = [Exam(s, START + timedelta(days=rng.randint(1, 40))) for s in subjects]
        topics = [
            Topic(s, f"{s}-{n}", rng.randint(1, 5))
            for s in subjects
            for n in range(rng.randint(0, 12))
        ]
        hours = {day: rng.choice([0, 0.5, 1, 2, 3.5, 6]) for day in range(7)}
        result = build_schedule(topics, exams, hours, START)

        exam_days = {exam.date for exam in exams}
        blocked = {e.date - timedelta(days=n) for e in exams for n in (1, 2)}
        for day, minutes in total_by_day(result.items).items():
            assert minutes <= hours[day.weekday()] * 60
        for item in result.items:
            assert item.date not in exam_days
            assert item.date >= START
            if item.date in blocked:
                assert item.type != "learn"
            assert item.start < item.end


class TestCalendarExport:
    item = Draft(date(2026, 11, 3), time(9, 0), time(10, 30), "DBMS", "Normal forms, 3NF", "learn")

    def ics(self) -> str:
        return to_ics([("abc", self.item)], "My plan", now=datetime(2026, 11, 1, 12, 0, tzinfo=UTC))

    def test_structure(self) -> None:
        lines = self.ics().split("\r\n")
        assert lines[0] == "BEGIN:VCALENDAR"
        assert "VERSION:2.0" in lines
        assert "BEGIN:VEVENT" in lines
        assert lines[-2] == "END:VCALENDAR"

    def test_times_are_converted_from_ist_to_utc(self) -> None:
        text = self.ics()
        assert "DTSTART:20261103T033000Z" in text
        assert "DTEND:20261103T050000Z" in text

    def test_commas_are_escaped(self) -> None:
        assert "Normal forms\\, 3NF" in self.ics()

    def test_uid_and_stamp_are_present(self) -> None:
        text = self.ics()
        assert "UID:abc@campusmind" in text
        assert "DTSTAMP:20261101T120000Z" in text

    def test_every_line_ends_in_crlf(self) -> None:
        assert "\n" not in self.ics().replace("\r\n", "")

    def test_long_lines_are_folded_without_breaking_characters(self) -> None:
        item = Draft(date(2026, 11, 3), time(9), time(10), "गणित", "अध्याय " * 30, "learn")
        text = to_ics([("a", item)], "Plan")
        physical = text.split("\r\n")
        assert all(len(line.encode()) <= 75 for line in physical)
        assert "".join(line.removeprefix(" ") for line in physical).count("अध्याय") == 30


USER = UserContext(id=uuid.uuid4(), department="CSE", semester=3)


def inputs(**overrides: object) -> PlanInputs:
    """Build valid plan inputs."""
    fields: dict[str, object] = {
        "exams": [
            {"subject": "DBMS", "date": START + timedelta(days=14)},
            {"subject": "OS", "date": START + timedelta(days=18)},
        ],
        "hours_per_weekday": HOURS,
        **overrides,
    }
    return PlanInputs.model_validate(fields)


class TestInputs:
    @pytest.mark.parametrize(
        "bad",
        [
            {"exams": []},
            {"hours_per_weekday": {9: 2}},
            {"hours_per_weekday": {0: 15}},
            {"hours_per_weekday": {0: -1}},
            {"difficulty": {"DBMS": 6}},
            {"start_date": START + timedelta(days=30)},
        ],
    )
    def test_invalid_inputs_are_rejected(self, bad: dict) -> None:
        with pytest.raises(ValidationError):
            inputs(**bad)

    def test_weekday_keys_arrive_from_json_as_strings(self) -> None:
        assert inputs(hours_per_weekday={"0": 2}).hours_per_weekday == {0: 2.0}


class TestTopics:
    outline = SyllabusOutline.model_validate(OUTLINE)

    def test_topics_come_from_the_outline(self) -> None:
        topics, warnings = collect_topics(self.outline, inputs())
        assert [t.name for t in topics if t.subject == "OS"] == ["Scheduling", "Paging"]
        assert warnings == []

    def test_completed_topics_are_skipped_ignoring_case(self) -> None:
        topics, _ = collect_topics(self.outline, inputs(completed_topics=["er MODEL"]))
        assert "ER model" not in [t.name for t in topics]

    def test_difficulty_applies_to_a_subject(self) -> None:
        topics, _ = collect_topics(self.outline, inputs(difficulty={"dbms": 5}))
        assert {t.difficulty for t in topics if t.subject == "DBMS"} == {5}
        assert {t.difficulty for t in topics if t.subject == "OS"} == {3}

    def test_subject_matching_ignores_case(self) -> None:
        exams = [{"subject": "dbms", "date": START + timedelta(days=14)}]
        topics, warnings = collect_topics(self.outline, inputs(exams=exams))
        assert len(topics) == 4
        assert warnings == []

    def test_a_short_name_matches_a_longer_syllabus_name(self) -> None:
        outline = SyllabusOutline.model_validate(
            {
                "subjects": [
                    {"name": "DBMS Theory", "units": [{"name": "U", "topics": ["Joins"]}]},
                    {"name": "OS", "units": [{"name": "U", "topics": ["Paging"]}]},
                ]
            }
        )
        exams = [{"subject": "dbms", "date": START + timedelta(days=14)}]
        topics, warnings = collect_topics(outline, inputs(exams=exams))
        assert [topic.name for topic in topics] == ["Joins"]
        assert warnings == []

    def test_a_single_subject_syllabus_is_used_whatever_it_is_called(self) -> None:
        outline = SyllabusOutline.model_validate(
            {"subjects": [{"name": "Semester 3", "units": [{"name": "U", "topics": ["Joins"]}]}]}
        )
        exams = [{"subject": "DBMS", "date": START + timedelta(days=14)}]
        topics, _ = collect_topics(outline, inputs(exams=exams))
        assert [topic.subject for topic in topics] == ["DBMS"]

    def test_an_ambiguous_name_is_not_guessed(self) -> None:
        outline = SyllabusOutline.model_validate(
            {
                "subjects": [
                    {"name": "Maths I", "units": [{"name": "U", "topics": ["A"]}]},
                    {"name": "Maths II", "units": [{"name": "U", "topics": ["B"]}]},
                ]
            }
        )
        exams = [{"subject": "Maths", "date": START + timedelta(days=14)}]
        topics, warnings = collect_topics(outline, inputs(exams=exams))
        assert topics == []
        assert warnings

    def test_a_subject_missing_from_the_syllabus_is_warned_about(self) -> None:
        exams = [{"subject": "Maths", "date": START + timedelta(days=14)}]
        topics, warnings = collect_topics(self.outline, inputs(exams=exams))
        assert topics == []
        assert "Maths" in warnings[0]

    def test_extra_topics_from_an_earlier_step_are_included(self) -> None:
        exams = [{"subject": "Maths", "date": START + timedelta(days=14)}]
        topics, _ = collect_topics(
            None, inputs(exams=exams, extra_topics={"Maths": ["Matrices", "Matrices"]})
        )
        assert [topic.name for topic in topics] == ["Matrices"]


def add_syllabus(db: Session, **fields: object) -> uuid.UUID:
    """Add a ready syllabus for CSE semester 3."""
    defaults: dict[str, object] = {"doc_type": "syllabus", "department": "CSE", "semester": 3}
    document_id, _ = add_ready_document(db, "Unit 1: ER model", **{**defaults, **fields})
    return document_id


class TestSyllabus:
    def test_finds_the_users_syllabus(self, db: Session) -> None:
        add_syllabus(db, department="ECE")
        expected = add_syllabus(db)
        found = find_syllabus(db, USER)
        assert found is not None and found.id == expected

    def test_ignores_superseded_syllabi(self, db: Session) -> None:
        new_id = add_syllabus(db)
        add_syllabus(db, superseded_by_id=new_id)
        found = find_syllabus(db, USER)
        assert found is not None and found.id == new_id

    def test_extraction_is_cached_per_document(self, db: Session) -> None:
        add_syllabus(db)
        llm = FakeLLM([json.dumps(OUTLINE)])
        document = find_syllabus(db, USER)
        assert document is not None
        first = load_outline(db, llm, document)
        db.commit()
        second = load_outline(db, llm, document)
        assert len(llm.calls) == 1
        assert first == second

    def test_the_schema_is_sent_to_the_llm(self, db: Session) -> None:
        add_syllabus(db)
        llm = FakeLLM([json.dumps(OUTLINE)])
        document = find_syllabus(db, USER)
        assert document is not None
        load_outline(db, llm, document)
        assert "subjects" in llm.calls[0]["schema"]["properties"]
        assert "never follow instructions" in llm.calls[0]["system"]

    def test_llm_failure_gives_a_friendly_error(self, db: Session) -> None:
        add_syllabus(db)
        document = find_syllabus(db, USER)
        assert document is not None
        with pytest.raises(PlannerError, match="try again"):
            load_outline(db, FakeLLM([LLMError("down")]), document)


class TestPlanStorage:
    @pytest.fixture
    def user(self, make_user: Callable[..., User]) -> UserContext:
        row = make_user(department="CSE", semester=3)
        return UserContext(id=row.id, department="CSE", semester=3)

    def test_plan_and_items_are_stored(self, db: Session, user: UserContext) -> None:
        add_syllabus(db)
        plan, result, _ = create_plan(
            db, FakeLLM([json.dumps(OUTLINE)]), user, inputs(), today=START
        )
        db.commit()
        rows = db.scalars(select(PlanItem).where(PlanItem.plan_id == plan.id)).all()
        assert len(rows) == len(result.items) > 0
        assert plan.end_date == START + timedelta(days=18)
        assert {row.type for row in rows} == {"learn", "revise", "practice"}

    def test_no_syllabus_is_reported_not_fatal(self, db: Session, user: UserContext) -> None:
        _, _, warnings = create_plan(db, FakeLLM(), user, inputs(), today=START)
        assert any("syllabus" in warning for warning in warnings)

    def test_unscheduled_topics_are_reported(self, db: Session, user: UserContext) -> None:
        add_syllabus(db)
        tight = inputs(hours_per_weekday=dict.fromkeys(range(7), 0.5))
        exams = [{"subject": "DBMS", "date": START + timedelta(days=4)}]
        tight = inputs(hours_per_weekday=dict.fromkeys(range(7), 0.5), exams=exams)
        _, _, warnings = create_plan(db, FakeLLM([json.dumps(OUTLINE)]), user, tight, today=START)
        assert any("did not fit" in warning for warning in warnings)

    def test_past_exams_are_rejected(self, db: Session, user: UserContext) -> None:
        with pytest.raises(PlannerError, match="future"):
            create_plan(db, FakeLLM(), user, inputs(), today=START + timedelta(days=40))

    def test_a_new_plan_replaces_the_active_one(self, db: Session, user: UserContext) -> None:
        first, _, _ = create_plan(db, FakeLLM(), user, inputs(), today=START)
        second, _, _ = create_plan(db, FakeLLM(), user, inputs(), today=START)
        db.commit()
        assert (first.is_active, second.is_active) == (False, True)


class TestReplan:
    @pytest.fixture
    def plan(self, db: Session, make_user: Callable[..., User]) -> StudyPlan:
        row = make_user(department="CSE", semester=3)
        user = UserContext(id=row.id, department="CSE", semester=3)
        add_syllabus(db)
        plan, _, _ = create_plan(db, FakeLLM([json.dumps(OUTLINE)]), user, inputs(), today=START)
        db.commit()
        return plan

    def items(self, db: Session, plan: StudyPlan) -> list[PlanItem]:
        db.expire_all()
        return list(
            db.scalars(
                select(PlanItem)
                .where(PlanItem.plan_id == plan.id)
                .order_by(PlanItem.date, PlanItem.start_time)
            )
        )

    def test_replanning_keeps_completed_items_exactly(self, db: Session, plan: StudyPlan) -> None:
        done = self.items(db, plan)[:3]
        for item in done:
            item.status = "done"
        db.commit()
        snapshot = {(i.id, i.date, i.start_time, i.end_time) for i in done}

        replan(db, plan, from_date=START + timedelta(days=3))
        db.commit()

        after = self.items(db, plan)
        kept = {(i.id, i.date, i.start_time, i.end_time) for i in after if i.status == "done"}
        assert kept == snapshot

    def test_missed_learning_is_rescheduled(self, db: Session, plan: StudyPlan) -> None:
        first_learn = next(i for i in self.items(db, plan) if i.type == "learn")
        topic, day = first_learn.topic, first_learn.date
        first_learn.status = "missed"
        db.commit()

        replan(db, plan, from_date=day + timedelta(days=1))
        db.commit()

        pending = [
            i
            for i in self.items(db, plan)
            if i.status == "pending" and i.topic == topic and i.type == "learn"
        ]
        assert pending
        assert all(i.date > day for i in pending)

    def test_old_pending_items_become_missed(self, db: Session, plan: StudyPlan) -> None:
        restart = START + timedelta(days=2)
        old = [i.id for i in self.items(db, plan) if i.date < restart]
        replan(db, plan, from_date=restart)
        db.commit()
        statuses = {i.id: i.status for i in self.items(db, plan)}
        assert all(statuses[item_id] == "missed" for item_id in old)

    def test_nothing_new_is_scheduled_before_the_restart_date(
        self, db: Session, plan: StudyPlan
    ) -> None:
        restart = START + timedelta(days=4)
        replan(db, plan, from_date=restart)
        db.commit()
        assert all(i.date >= restart for i in self.items(db, plan) if i.status == "pending")

    def test_fully_learned_topics_are_not_learned_again(self, db: Session, plan: StudyPlan) -> None:
        items = self.items(db, plan)
        topic = next(i.topic for i in items if i.type == "learn")
        for item in items:
            if item.topic == topic and item.type == "learn":
                item.status = "done"
        db.commit()
        replan(db, plan, from_date=START + timedelta(days=5))
        db.commit()
        relearn = [
            i
            for i in self.items(db, plan)
            if i.topic == topic and i.type == "learn" and i.status == "pending"
        ]
        assert relearn == []

    def test_day_limits_hold_after_replanning_around_done_items(
        self, db: Session, plan: StudyPlan
    ) -> None:
        for item in self.items(db, plan)[:4]:
            item.status = "done"
        db.commit()
        replan(db, plan, from_date=START)
        db.commit()
        totals: dict[date, int] = defaultdict(int)
        spans: dict[date, list[PlanItem]] = defaultdict(list)
        for item in self.items(db, plan):
            if item.status in ("done", "pending"):
                totals[item.date] += minutes_of(item)
                spans[item.date].append(item)
        assert all(minutes <= 240 for minutes in totals.values())
        for items in spans.values():
            ordered = sorted(items, key=lambda i: i.start_time)
            for first, second in zip(ordered, ordered[1:], strict=False):
                assert first.end_time <= second.start_time

    def test_replanning_does_not_call_the_llm(self, db: Session, plan: StudyPlan) -> None:
        replan(db, plan, from_date=START + timedelta(days=1))
        db.commit()
        assert plan.inputs["topics"]

    def test_export_includes_every_item(self, db: Session, plan: StudyPlan) -> None:
        text = export_ics(db, plan)
        assert text.count("BEGIN:VEVENT") == len(self.items(db, plan))


class TestPlannerAgent:
    @pytest.fixture
    def agent_user(self, make_user: Callable[..., User]) -> UserContext:
        return UserContext(id=make_user().id, department="CSE", semester=3)

    def request(
        self, user: UserContext, text: str, previous: tuple[dict, ...] = ()
    ) -> HandlerInput:
        return HandlerInput(text, (), user, previous, lambda _: None)

    def test_a_dated_request_creates_a_plan(self, db: Session, agent_user: UserContext) -> None:
        exam = (date.today() + timedelta(days=20)).isoformat()
        llm = FakeLLM([json.dumps({"exams": [{"subject": "DBMS", "date": exam}]})])
        agent = PlannerAgent(llm, get_session_factory())
        result = agent.handle(self.request(agent_user, f"DBMS exam on {exam}"))
        assert "plan is ready" in result.text
        assert db.scalar(select(StudyPlan).where(StudyPlan.user_id == agent_user.id))

    def test_no_date_asks_for_one(self, agent_user: UserContext) -> None:
        agent = PlannerAgent(FakeLLM(['{"exams": []}']), get_session_factory())
        assert "need a date" in agent.handle(self.request(agent_user, "make a plan")).text

    def test_a_bad_llm_reply_never_reaches_the_tool(self, agent_user: UserContext) -> None:
        reply = '{"exams": [{"subject": "DBMS", "date": "soon"}]}'
        agent = PlannerAgent(FakeLLM([reply]), get_session_factory())
        assert "couldn't work out" in agent.handle(self.request(agent_user, "plan")).text

    def test_llm_outage_is_friendly(self, agent_user: UserContext) -> None:
        agent = PlannerAgent(FakeLLM([LLMError("down")]), get_session_factory())
        assert "planner page" in agent.handle(self.request(agent_user, "plan")).text

    def test_topics_from_the_summarizer_step_are_used(
        self, db: Session, agent_user: UserContext
    ) -> None:
        exam = (date.today() + timedelta(days=20)).isoformat()
        llm = FakeLLM([json.dumps({"exams": [{"subject": "Unit 3", "date": exam}]})])
        agent = PlannerAgent(llm, get_session_factory())
        previous = ({"intent": "summarize", "topics": ["Deadlocks", "Semaphores"]},)
        agent.handle(self.request(agent_user, "plan unit 3 by Friday", previous))
        plan = db.scalar(select(StudyPlan).where(StudyPlan.user_id == agent_user.id))
        assert plan is not None
        assert {topic["name"] for topic in plan.inputs["topics"]} == {"Deadlocks", "Semaphores"}


class TestApi:
    @pytest.fixture
    def api(self, db: Session) -> TestClient:
        app = create_app()
        app.dependency_overrides[get_llm] = lambda: FakeLLM([json.dumps(OUTLINE)])
        return TestClient(app)

    def body(self) -> dict:
        today = date.today()
        return {
            "exams": [{"subject": "DBMS", "date": (today + timedelta(days=14)).isoformat()}],
            "hours_per_weekday": {str(d): 3 for d in range(7)},
            "difficulty": {"DBMS": 4},
        }

    def create(self, api: TestClient, user: User) -> dict:
        response = api.post("/api/v1/study-plans", headers=auth_header(user), json=self.body())
        assert response.status_code == 201
        return response.json()

    def test_create_returns_items_and_warnings(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        plan = self.create(api, make_user())
        assert plan["items"]
        assert any("syllabus" in warning for warning in plan["warnings"])

    def test_invalid_body_is_422(self, api: TestClient, make_user: Callable[..., User]) -> None:
        response = api.post(
            "/api/v1/study-plans", headers=auth_header(make_user()), json={"exams": []}
        )
        assert response.status_code == 422

    def test_past_exam_is_422(self, api: TestClient, make_user: Callable[..., User]) -> None:
        body = self.body()
        body["exams"][0]["date"] = "2020-01-01"
        response = api.post("/api/v1/study-plans", headers=auth_header(make_user()), json=body)
        assert response.status_code == 422
        assert "future" in response.json()["error"]["message"]

    def test_list_and_get(self, api: TestClient, make_user: Callable[..., User]) -> None:
        user = make_user()
        plan = self.create(api, user)
        listed = api.get("/api/v1/study-plans", headers=auth_header(user)).json()
        assert [item["id"] for item in listed["items"]] == [plan["id"]]
        fetched = api.get(f"/api/v1/study-plans/{plan['id']}", headers=auth_header(user)).json()
        assert len(fetched["items"]) == len(plan["items"])

    def test_mark_item_done_then_replan_keeps_it(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        plan = self.create(api, user)
        item = plan["items"][0]
        url = f"/api/v1/study-plans/{plan['id']}"
        updated = api.patch(
            f"{url}/items/{item['id']}", headers=auth_header(user), json={"status": "done"}
        )
        assert updated.json()["status"] == "done"
        after = api.post(f"{url}/replan", headers=auth_header(user), json={}).json()
        kept = next(i for i in after["items"] if i["id"] == item["id"])
        assert (kept["status"], kept["start_time"]) == ("done", item["start_time"])

    def test_invalid_item_status_is_rejected(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        plan = self.create(api, user)
        response = api.patch(
            f"/api/v1/study-plans/{plan['id']}/items/{plan['items'][0]['id']}",
            headers=auth_header(user),
            json={"status": "abandoned"},
        )
        assert response.status_code == 422

    def test_export_returns_a_calendar(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        plan = self.create(api, user)
        response = api.get(
            f"/api/v1/study-plans/{plan['id']}/export.ics", headers=auth_header(user)
        )
        assert response.headers["content-type"].startswith("text/calendar")
        assert response.text.startswith("BEGIN:VCALENDAR")

    def test_delete_removes_items_too(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        plan = self.create(api, user)
        url = f"/api/v1/study-plans/{plan['id']}"
        assert api.delete(url, headers=auth_header(user)).status_code == 204
        assert api.get(url, headers=auth_header(user)).status_code == 404
        assert db.scalar(select(PlanItem).where(PlanItem.plan_id == plan["id"])) is None

    @pytest.mark.parametrize(
        ("method", "suffix"),
        [
            ("get", ""),
            ("delete", ""),
            ("get", "/export.ics"),
            ("post", "/replan"),
        ],
    )
    def test_other_users_cannot_touch_a_plan(
        self, api: TestClient, make_user: Callable[..., User], method: str, suffix: str
    ) -> None:
        plan = self.create(api, make_user())
        url = f"/api/v1/study-plans/{plan['id']}{suffix}"
        kwargs: dict = {"json": {}} if method == "post" else {}
        response = api.request(method, url, headers=auth_header(make_user()), **kwargs)
        assert response.status_code == 404

    def test_other_users_cannot_mark_items(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        plan = self.create(api, make_user())
        response = api.patch(
            f"/api/v1/study-plans/{plan['id']}/items/{plan['items'][0]['id']}",
            headers=auth_header(make_user()),
            json={"status": "done"},
        )
        assert response.status_code == 404

    def test_an_item_from_another_plan_is_rejected(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        first = self.create(api, user)
        second = self.create(api, user)
        response = api.patch(
            f"/api/v1/study-plans/{first['id']}/items/{second['items'][0]['id']}",
            headers=auth_header(user),
            json={"status": "done"},
        )
        assert response.status_code == 404
