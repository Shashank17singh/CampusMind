"""The study schedule algorithm, and .ics export.

This module is pure Python with no LLM and no database. The same input always gives the
same plan, so the hard rules below can be proven by tests rather than hoped for:

- A day never exceeds its available hours.
- The two days before an exam hold revision only.
- Nothing is scheduled on an exam day.
- Each topic is learned before its subject's revision window starts.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

SLOT_MINUTES = 30
REVISE_MINUTES = 30
PRACTICE_MINUTES = 60
REVISION_OFFSETS = (1, 3, 7)
REVISION_ONLY_DAYS = 2
# Learning stops at 75% of a day, so that spaced revisions have room to land later.
LEARN_SHARE = 0.75
LEARN_MINUTES = {1: 60, 2: 60, 3: 90, 4: 120, 5: 150}
DEFAULT_DAY_START = time(9, 0)
PRACTICE_TOPIC = "Past papers"
GENERAL_TOPIC = "General revision"


@dataclass(frozen=True)
class Exam:
    """A subject's exam date."""

    subject: str
    date: date


@dataclass(frozen=True)
class Topic:
    """Something to learn."""

    subject: str
    name: str
    difficulty: int = 3


@dataclass(frozen=True)
class Draft:
    """A scheduled slot, before it is stored."""

    date: date
    start: time
    end: time
    subject: str
    topic: str
    type: str


@dataclass
class Busy:
    """Time on a day that is already taken, by completed items during a re-plan."""

    minutes: int = 0
    until: time = DEFAULT_DAY_START


@dataclass
class ScheduleResult:
    """The schedule, and any topics that could not fit."""

    items: list[Draft] = field(default_factory=list)
    unscheduled: list[Topic] = field(default_factory=list)


@dataclass
class _Block:
    """Minutes of one activity on one day, before clock times are assigned."""

    subject: str
    topic: str
    type: str
    minutes: int


def _round_down(minutes: int) -> int:
    """Round down to a whole number of slots."""
    return minutes - minutes % SLOT_MINUTES


def _add_minutes(start: time, minutes: int) -> time:
    """Add minutes to a clock time within one day."""
    return (datetime.combine(date.min, start) + timedelta(minutes=minutes)).time()


class _Scheduler:
    """Holds the working state of one scheduling run."""

    def __init__(
        self,
        topics: list[Topic],
        exams: list[Exam],
        hours_by_weekday: dict[int, float],
        start: date,
        busy: dict[date, Busy],
        learned_on: dict[Topic, date],
    ) -> None:
        self.topics = topics
        self.exams = sorted(exams, key=lambda exam: (exam.date, exam.subject))
        self.exam_by_subject = {exam.subject: exam.date for exam in self.exams}
        self.start = start
        self.learned_on = dict(learned_on)
        self.blocks: dict[date, list[_Block]] = defaultdict(list)
        self.exam_days = {exam.date for exam in self.exams}
        self.revision_only = {
            exam.date - timedelta(days=offset)
            for exam in self.exams
            for offset in range(1, REVISION_ONLY_DAYS + 1)
        } - self.exam_days

        last = max(self.exam_days)
        self.days = [
            start + timedelta(days=offset)
            for offset in range((last - start).days)
            if start + timedelta(days=offset) not in self.exam_days
        ]
        self.remaining = {
            day: max(
                0,
                _round_down(int(hours_by_weekday.get(day.weekday(), 0) * 60))
                - busy.get(day, Busy()).minutes,
            )
            for day in self.days
        }
        self.learn_budget = {
            day: _round_down(int(self.remaining[day] * LEARN_SHARE)) for day in self.days
        }

    def _place(self, day: date, subject: str, topic: str, kind: str, minutes: int) -> None:
        """Record a block and use up the day's time."""
        self.blocks[day].append(_Block(subject, topic, kind, minutes))
        self.remaining[day] -= minutes

    def _priority(self, topic: Topic) -> tuple[float, date, str, str]:
        """Sort key: harder topics with closer exams come first."""
        exam = self.exam_by_subject[topic.subject]
        days_left = max((exam - self.start).days, 1)
        return (-topic.difficulty / days_left, exam, topic.subject, topic.name)

    def _learn(self, topic: Topic) -> bool:
        """Place a topic in the earliest free time, splitting across days if needed."""
        needed = LEARN_MINUTES[topic.difficulty]
        latest = self.exam_by_subject[topic.subject] - timedelta(days=REVISION_ONLY_DAYS + 1)
        placements: list[tuple[date, int]] = []
        for day in self.days:
            if day > latest or needed == 0:
                break
            if day in self.revision_only:
                continue
            taken = min(needed, self.learn_budget[day])
            if taken >= SLOT_MINUTES:
                placements.append((day, taken))
                needed -= taken

        if needed > 0:
            return False
        for day, minutes in placements:
            self.learn_budget[day] -= minutes
            self._place(day, topic.subject, topic.name, "learn", minutes)
        self.learned_on[topic] = placements[-1][0]
        return True

    def _add_spaced_revisions(self) -> None:
        """Revise each topic 1, 3 and 7 days after learning it, where those dates allow."""
        ordered = sorted(self.learned_on.items(), key=lambda p: (p[1], p[0].subject, p[0].name))
        for topic, learned in ordered:
            exam = self.exam_by_subject.get(topic.subject)
            if exam is None:
                continue
            for offset in REVISION_OFFSETS:
                target = learned + timedelta(days=offset)
                for day in (target, target + timedelta(days=1)):
                    if (
                        day < exam
                        and day in self.remaining
                        and day not in self.revision_only
                        and self.remaining[day] >= REVISE_MINUTES
                    ):
                        self._place(day, topic.subject, topic.name, "revise", REVISE_MINUTES)
                        break

    def _exam_for_revision_day(self, day: date) -> Exam:
        """Return the next exam after a revision-only day."""
        return next(exam for exam in self.exams if exam.date > day)

    def _fill_revision_days(self) -> None:
        """Fill revision-only days with past papers, then the subject's hardest topics."""
        practised: set[str] = set()
        cursors: dict[str, int] = defaultdict(int)
        for day in sorted(self.revision_only):
            if day not in self.remaining:
                continue
            subject = self._exam_for_revision_day(day).subject
            if subject not in practised and self.remaining[day] >= PRACTICE_MINUTES:
                self._place(day, subject, PRACTICE_TOPIC, "practice", PRACTICE_MINUTES)
                practised.add(subject)

            names = [
                topic.name
                for topic in sorted(
                    (t for t in self.topics if t.subject == subject),
                    key=lambda t: (-t.difficulty, t.name),
                )
            ] or [GENERAL_TOPIC]
            while self.remaining[day] >= REVISE_MINUTES:
                name = names[cursors[subject] % len(names)]
                cursors[subject] += 1
                self._place(day, subject, name, "revise", REVISE_MINUTES)

    def _clock_times(self, busy: dict[date, Busy], day_start: time) -> list[Draft]:
        """Lay each day's blocks out end to end, joining adjacent matching blocks."""
        drafts = []
        for day in sorted(self.blocks):
            kept = busy.get(day)
            cursor = max(day_start, kept.until) if kept else day_start
            merged: list[_Block] = []
            for block in sorted(self.blocks[day], key=lambda b: b.type == "learn"):
                last = merged[-1] if merged else None
                same = last and (last.subject, last.topic, last.type) == (
                    block.subject,
                    block.topic,
                    block.type,
                )
                if last and same:
                    last.minutes += block.minutes
                else:
                    merged.append(_Block(block.subject, block.topic, block.type, block.minutes))
            for block in merged:
                end = _add_minutes(cursor, block.minutes)
                drafts.append(Draft(day, cursor, end, block.subject, block.topic, block.type))
                cursor = end
        return drafts

    def run(self, busy: dict[date, Busy], day_start: time) -> ScheduleResult:
        """Run the phases in order."""
        known = [
            topic
            for topic in self.topics
            if topic.subject in self.exam_by_subject and topic not in self.learned_on
        ]
        unscheduled = [
            topic for topic in sorted(known, key=self._priority) if not self._learn(topic)
        ]
        self._add_spaced_revisions()
        self._fill_revision_days()
        return ScheduleResult(self._clock_times(busy, day_start), unscheduled)


def build_schedule(
    topics: list[Topic],
    exams: list[Exam],
    hours_by_weekday: dict[int, float],
    start: date,
    busy: dict[date, Busy] | None = None,
    learned_on: dict[Topic, date] | None = None,
    day_start: time = DEFAULT_DAY_START,
) -> ScheduleResult:
    """Build a schedule.

    `hours_by_weekday` maps Monday=0 to Sunday=6. `learned_on` gives topics already learned,
    so a re-plan can still schedule their revisions.
    """
    upcoming = [exam for exam in exams if exam.date > start]
    if not upcoming:
        return ScheduleResult(unscheduled=list(topics))
    occupied = busy or {}
    scheduler = _Scheduler(topics, upcoming, hours_by_weekday, start, occupied, learned_on or {})
    return scheduler.run(occupied, day_start)


def _escape_ics(text: str) -> str:
    """Escape a value for an iCalendar text field."""
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold_ics(line: str) -> str:
    """Fold a content line at 75 octets, as RFC 5545 requires."""
    encoded = line.encode()
    if len(encoded) <= 75:
        return line
    pieces: list[str] = []
    while encoded:
        limit = 75 if not pieces else 74
        cut = min(limit, len(encoded))
        # Never cut through the middle of a multi-byte character.
        while cut < len(encoded) and (encoded[cut] & 0xC0) == 0x80:
            cut -= 1
        pieces.append(encoded[:cut].decode())
        encoded = encoded[cut:]
    return "\r\n ".join(pieces)


def to_ics(
    items: list[tuple[str, Draft]],
    title: str,
    timezone: str = "Asia/Kolkata",
    now: datetime | None = None,
) -> str:
    """Render (uid, item) pairs as an iCalendar file, with all times in UTC."""
    zone = ZoneInfo(timezone)
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")

    def utc(day: date, moment: time) -> str:
        local = datetime.combine(day, moment, tzinfo=zone)
        return local.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//CampusMind//Study Planner//EN",
        "CALSCALE:GREGORIAN",
        f"X-WR-CALNAME:{_escape_ics(title)}",
    ]
    for uid, item in items:
        summary = f"{item.type.title()}: {item.subject} - {item.topic}"
        lines.extend(
            [
                "BEGIN:VEVENT",
                f"UID:{uid}@campusmind",
                f"DTSTAMP:{stamp}",
                f"DTSTART:{utc(item.date, item.start)}",
                f"DTEND:{utc(item.date, item.end)}",
                f"SUMMARY:{_escape_ics(summary)}",
                "END:VEVENT",
            ]
        )
    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold_ics(line) for line in lines) + "\r\n"
