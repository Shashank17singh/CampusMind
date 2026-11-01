"""Admin analytics endpoints. Each report can also be downloaded as CSV."""

import csv
import io
from collections.abc import Sequence
from datetime import date, timedelta
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from app.analytics import service
from app.analytics.service import DateRange
from app.auth.deps import DbSession, require_role
from app.models import User
from app.vectorstore.embeddings import Embedder, get_embedder

router = APIRouter(prefix="/admin/analytics", tags=["analytics"])

DEFAULT_RANGE_DAYS = 30
Format = Literal["json", "csv"]


def date_range(
    start: Annotated[date | None, Query(alias="from")] = None,
    end: Annotated[date | None, Query(alias="to")] = None,
) -> DateRange:
    """Read the date filter. The default is the last 30 days."""
    last = end or date.today()
    first = start or last - timedelta(days=DEFAULT_RANGE_DAYS - 1)
    if first > last:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail="`from` must not be after `to`"
        )
    return DateRange(first, last)


Period = Annotated[DateRange, Depends(date_range)]
Admin = Annotated[User, Depends(require_role("admin"))]
EmbedderDep = Annotated[Embedder, Depends(get_embedder)]


def _csv_cell(value: Any) -> Any:
    """Render a cell, and stop a spreadsheet treating text as a formula."""
    if isinstance(value, list):
        value = " | ".join(str(item) for item in value)
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t", "\r"):
        return f"'{value}"
    return value


def csv_response(rows: Sequence[dict[str, Any]], name: str) -> Response:
    """Turn report rows into a CSV download."""
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_cell(value) for key, value in row.items()})
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{name}.csv"'},
    )


def _report(rows: list[dict[str, Any]], name: str, output: Format) -> Any:
    """Return rows as JSON or CSV."""
    return csv_response(rows, name) if output == "csv" else rows


@router.get("/overview")
def overview(db: DbSession, _: Admin, period: Period, format: Format = "json") -> Any:  # noqa: A002
    """Return daily activity, average latency and the feedback ratio."""
    report = service.overview(db, period)
    if format == "csv":
        return csv_response(report["daily"], "daily-activity")
    return report


@router.get("/top-questions")
def top_questions(
    db: DbSession,
    _: Admin,
    period: Period,
    embedder: EmbedderDep,
    format: Format = "json",  # noqa: A002
) -> Any:
    """Return the 20 most asked questions, with similar wordings merged."""
    return _report(service.top_questions(db, period, embedder), "top-questions", format)


@router.get("/unanswered")
def unanswered(
    db: DbSession,
    _: Admin,
    period: Period,
    format: Format = "json",  # noqa: A002
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> Any:
    """List questions the assistant could not answer."""
    rows, total = service.unanswered_questions(db, period, limit, offset)
    if format == "csv":
        return csv_response(rows, "unanswered")
    return {"items": rows, "total": total}


@router.get("/intents")
def intents(db: DbSession, _: Admin, period: Period, format: Format = "json") -> Any:  # noqa: A002
    """Return how often each intent occurred."""
    return _report(service.intent_distribution(db, period), "intents", format)


@router.get("/top-documents")
def top_documents(db: DbSession, _: Admin, period: Period, format: Format = "json") -> Any:  # noqa: A002
    """Rank documents by citations."""
    return _report(service.top_documents(db, period), "top-documents", format)


@router.get("/cost")
def cost(db: DbSession, _: Admin, period: Period, format: Format = "json") -> Any:  # noqa: A002
    """Return token use and estimated LLM cost per day."""
    return _report(service.token_cost(db, period), "llm-cost", format)


@router.get("/knowledge-gaps")
def knowledge_gaps(
    db: DbSession,
    _: Admin,
    period: Period,
    embedder: EmbedderDep,
    format: Format = "json",  # noqa: A002
) -> Any:
    """Group unanswered and thumbs-down questions by topic."""
    return _report(service.knowledge_gaps(db, period, embedder), "knowledge-gaps", format)


@router.get("/document-status")
def document_status(db: DbSession, _: Admin, format: Format = "json") -> Any:  # noqa: A002
    """Count documents per ingestion status."""
    return _report(service.document_status_counts(db), "document-status", format)
