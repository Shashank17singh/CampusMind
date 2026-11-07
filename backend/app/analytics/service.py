"""Analytics queries.

Everything is aggregated in SQL, so reports stay fast as `messages` grows. The queries and
the indexes they use are listed in docs/notes/analytics.md.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import cast, func, literal_column, select, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Session

from app.analytics.clustering import Cluster, cluster_questions
from app.models import Document, Feedback, Message
from app.vectorstore.embeddings import Embedder

# A fixed offset, because not every PostgreSQL build ships a timezone database. India has
# no daylight saving, so this is always correct. A college elsewhere must change it.
REPORT_UTC_OFFSET = timedelta(hours=5, minutes=30)
MAX_QUESTIONS_TO_CLUSTER = 2000
GAP_CLUSTER_THRESHOLD = 0.75

# US dollars per million tokens, as (input, output). Local models cost nothing here.
# These change, so check them against the provider's current price list.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


@dataclass(frozen=True)
class DateRange:
    """A span of whole days in the college's timezone, with both ends included."""

    start: date
    end: date

    @property
    def starts_at(self) -> datetime:
        """The first instant of the range."""
        return datetime.combine(self.start, time.min, timezone(REPORT_UTC_OFFSET)).astimezone(UTC)

    @property
    def ends_before(self) -> datetime:
        """The first instant after the range."""
        following = self.end + timedelta(days=1)
        return datetime.combine(following, time.min, timezone(REPORT_UTC_OFFSET)).astimezone(UTC)


def _in_range(column: Any, period: DateRange) -> tuple[Any, Any]:
    """Build a range filter that can use an index on the column."""
    return column >= period.starts_at, column < period.ends_before


def _local_day(column: Any) -> Any:
    """Convert a timestamp to a calendar day in the college's timezone."""
    return func.date(func.timezone(REPORT_UTC_OFFSET, column))


def daily_activity(session: Session, period: DateRange) -> list[dict[str, Any]]:
    """Return active users and questions per day."""
    day = _local_day(Message.created_at).label("day")
    rows = session.execute(
        select(
            day,
            func.count(func.distinct(Message.user_id)).label("active_users"),
            func.count().label("questions"),
        )
        .where(Message.role == "user", *_in_range(Message.created_at, period))
        .group_by(day)
        .order_by(day)
    ).all()
    return [dict(row._mapping) for row in rows]


def overview(session: Session, period: DateRange) -> dict[str, Any]:
    """Return the headline numbers."""
    average_latency = session.scalar(
        select(func.avg(Message.latency_ms)).where(
            Message.role == "assistant", *_in_range(Message.created_at, period)
        )
    )
    votes = session.execute(
        select(
            func.count().filter(Feedback.rating == 1).label("up"),
            func.count().filter(Feedback.rating == -1).label("down"),
        ).where(*_in_range(Feedback.created_at, period))
    ).one()
    total_votes = votes.up + votes.down
    return {
        "daily": daily_activity(session, period),
        "avg_latency_ms": round(float(average_latency), 1) if average_latency else None,
        "thumbs_up": votes.up,
        "thumbs_down": votes.down,
        "feedback_ratio": round(votes.up / total_votes, 3) if total_votes else None,
    }


def intent_distribution(session: Session, period: DateRange) -> list[dict[str, Any]]:
    """Count assistant messages per intent."""
    rows = session.execute(
        select(Message.intent, func.count().label("count"))
        .where(
            Message.role == "assistant",
            Message.intent.is_not(None),
            *_in_range(Message.created_at, period),
        )
        .group_by(Message.intent)
        .order_by(func.count().desc())
    ).all()
    return [dict(row._mapping) for row in rows]


def top_documents(session: Session, period: DateRange, limit: int = 20) -> list[dict[str, Any]]:
    """Rank documents by how often answers cite them."""
    citation = func.jsonb_array_elements(cast(Message.citations, JSONB)).table_valued(
        "value", name="citation"
    )
    document_id = cast(citation.c.value.op("->>")("document_id"), PgUUID(as_uuid=True))
    rows = session.execute(
        select(
            Document.id.label("document_id"),
            Document.title,
            func.count().label("citations"),
        )
        .select_from(Message)
        .join(citation, literal_column("true"))
        .join(Document, Document.id == document_id)
        .where(Message.role == "assistant", *_in_range(Message.created_at, period))
        .group_by(Document.id, Document.title)
        .order_by(func.count().desc())
        .limit(limit)
    ).all()
    return [dict(row._mapping) for row in rows]


def token_cost(session: Session, period: DateRange) -> list[dict[str, Any]]:
    """Return token use and estimated cost per day and model."""
    day = _local_day(Message.created_at).label("day")
    rows = session.execute(
        select(
            day,
            Message.model_name,
            func.coalesce(func.sum(Message.tokens_in), 0).label("tokens_in"),
            func.coalesce(func.sum(Message.tokens_out), 0).label("tokens_out"),
        )
        .where(Message.role == "assistant", *_in_range(Message.created_at, period))
        .group_by(day, Message.model_name)
        .order_by(day, Message.model_name)
    ).all()
    report = []
    for row in rows:
        price_in, price_out = MODEL_PRICES.get(row.model_name or "", (0.0, 0.0))
        cost = (row.tokens_in * price_in + row.tokens_out * price_out) / 1_000_000
        report.append({**row._mapping, "cost_usd": round(cost, 6)})
    return report


def _question_column() -> Any:
    """Build a subquery finding the user question an assistant message answered."""
    return (
        select(literal_column("q.content"))
        .select_from(text("messages AS q"))
        .where(
            text(
                "q.conversation_id = messages.conversation_id "
                "AND q.role = 'user' AND q.created_at <= messages.created_at"
            )
        )
        .order_by(text("q.created_at DESC"))
        .limit(1)
        .scalar_subquery()
    )


def unanswered_questions(
    session: Session, period: DateRange, limit: int = 100, offset: int = 0
) -> tuple[list[dict[str, Any]], int]:
    """List questions the assistant could not answer, newest first."""
    filters = [Message.answered.is_(False), *_in_range(Message.created_at, period)]
    total = session.scalar(select(func.count()).select_from(Message).where(*filters)) or 0
    rows = session.execute(
        select(
            _question_column().label("question"),
            Message.created_at.label("asked_at"),
            Message.user_id,
        )
        .where(*filters)
        .order_by(Message.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return [dict(row._mapping) for row in rows], total


def _asked_questions(session: Session, period: DateRange) -> Sequence[str]:
    """Return the most recent questions in the range."""
    return session.scalars(
        select(Message.content)
        .where(Message.role == "user", *_in_range(Message.created_at, period))
        .order_by(Message.created_at.desc())
        .limit(MAX_QUESTIONS_TO_CLUSTER)
    ).all()


def _rows_from(clusters: list[Cluster], limit: int, key: str) -> list[dict[str, Any]]:
    """Convert clusters to report rows."""
    return [
        {key: item.representative, "count": item.count, "examples": item.examples}
        for item in clusters[:limit]
    ]


def top_questions(
    session: Session, period: DateRange, embedder: Embedder, limit: int = 20
) -> list[dict[str, Any]]:
    """Return the most asked questions, merging those above 0.85 similarity."""
    clusters = cluster_questions(list(_asked_questions(session, period)), embedder, 0.85)
    return _rows_from(clusters, limit, "question")


def knowledge_gaps(
    session: Session, period: DateRange, embedder: Embedder, limit: int = 50
) -> list[dict[str, Any]]:
    """Group unanswered and disliked questions by topic, showing faculty what to upload."""
    disliked_or_unanswered = (
        select(_question_column().label("question"))
        .select_from(Message)
        .outerjoin(Feedback, (Feedback.message_id == Message.id) & (Feedback.rating == -1))
        .where(
            Message.role == "assistant",
            (Message.answered.is_(False)) | (Feedback.id.is_not(None)),
            *_in_range(Message.created_at, period),
        )
        .order_by(Message.created_at.desc())
        .limit(MAX_QUESTIONS_TO_CLUSTER)
    )
    questions = [row.question for row in session.execute(disliked_or_unanswered) if row.question]
    clusters = cluster_questions(questions, embedder, GAP_CLUSTER_THRESHOLD)
    return _rows_from(clusters, limit, "topic")


def document_status_counts(session: Session) -> list[dict[str, Any]]:
    """Count documents per ingestion status."""
    rows = session.execute(
        select(Document.status, func.count().label("count"))
        .group_by(Document.status)
        .order_by(Document.status)
    ).all()
    return [dict(row._mapping) for row in rows]
