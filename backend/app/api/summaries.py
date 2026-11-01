"""Summary endpoints."""

import uuid
from datetime import datetime
from pathlib import Path, PurePath
from typing import Annotated, Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select

from app.agents.summarizer import Mode, SummaryContent
from app.agents.summary_jobs import run_summary_task, to_markdown, to_pdf
from app.auth.deps import CurrentUser, DbSession
from app.config import get_settings
from app.ingestion.errors import FileTooLargeError
from app.ingestion.validation import validate_upload
from app.models import Document, Summary

router = APIRouter(prefix="/summaries", tags=["summaries"])


class SummaryRequest(BaseModel):
    """A request to summarize a stored document."""

    document_id: uuid.UUID
    mode: Mode


class SummaryAccepted(BaseModel):
    """The id to poll for a queued summary."""

    summary_id: uuid.UUID


class SummaryOut(BaseModel):
    """A summary."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    document_id: uuid.UUID | None
    source_name: str
    mode: str
    status: str
    error_message: str | None
    content: dict[str, Any] | None
    dropped_count: int
    tokens_in: int
    tokens_out: int
    duration_ms: int | None
    created_at: datetime


class SummaryPage(BaseModel):
    """One page of summaries."""

    items: list[SummaryOut]
    total: int


def _load(db: DbSession, summary_id: uuid.UUID, user: CurrentUser) -> Summary:
    """Load the user's own summary. Someone else's looks the same as a missing one."""
    summary = db.get(Summary, summary_id)
    if summary is None or summary.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Summary not found")
    return summary


@router.post("", response_model=SummaryAccepted, status_code=status.HTTP_202_ACCEPTED)
def summarize_document(body: SummaryRequest, db: DbSession, user: CurrentUser) -> SummaryAccepted:
    """Queue a summary of a document in the store."""
    document = db.get(Document, body.document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Document not found")
    if document.status != "ready":
        raise HTTPException(status.HTTP_409_CONFLICT, detail="This document is not ready yet")

    summary = Summary(
        user_id=user.id, document_id=document.id, source_name=document.title, mode=body.mode
    )
    db.add(summary)
    db.commit()
    run_summary_task.delay(str(summary.id))
    return SummaryAccepted(summary_id=summary.id)


@router.post("/upload", response_model=SummaryAccepted, status_code=status.HTTP_202_ACCEPTED)
def summarize_upload(
    db: DbSession,
    user: CurrentUser,
    file: Annotated[UploadFile, File()],
    mode: Annotated[Mode, Form()],
) -> SummaryAccepted:
    """Queue a summary of an uploaded file. The file is deleted once it has been read."""
    settings = get_settings()
    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise FileTooLargeError(f"The file is larger than the {settings.max_upload_mb} MB limit.")
    filename = PurePath(file.filename or "upload").name
    validate_upload(filename, data, settings.max_upload_bytes)

    summary = Summary(user_id=user.id, source_name=filename, mode=mode)
    db.add(summary)
    db.commit()

    folder = Path(settings.upload_dir) / "summaries"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{summary.id}{PurePath(filename).suffix.lower()}"
    path.write_bytes(data)
    run_summary_task.delay(str(summary.id), str(path))
    return SummaryAccepted(summary_id=summary.id)


@router.get("", response_model=SummaryPage)
def list_summaries(
    db: DbSession,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> SummaryPage:
    """List the user's summaries, newest first."""
    filters = [Summary.user_id == user.id]
    total = db.scalar(select(func.count()).select_from(Summary).where(*filters)) or 0
    rows = db.scalars(
        select(Summary)
        .where(*filters)
        .order_by(Summary.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return SummaryPage(items=[SummaryOut.model_validate(row) for row in rows], total=total)


@router.get("/{summary_id}", response_model=SummaryOut)
def get_summary(summary_id: uuid.UUID, db: DbSession, user: CurrentUser) -> Summary:
    """Return a summary and its status."""
    return _load(db, summary_id, user)


@router.get("/{summary_id}/export")
def export_summary(
    summary_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
    file_format: Annotated[Literal["md", "pdf"], Query(alias="format")] = "md",
) -> Response:
    """Download a finished summary as Markdown or PDF."""
    summary = _load(db, summary_id, user)
    if summary.status != "ready" or summary.content is None:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="This summary is not ready yet")

    content = SummaryContent.model_validate(summary.content)
    title = f"{summary.source_name}: {summary.mode.replace('_', ' ')}"
    if file_format == "pdf":
        body: str | bytes = to_pdf(title, content)
        media_type = "application/pdf"
    else:
        body = to_markdown(title, content)
        media_type = "text/markdown; charset=utf-8"
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="summary.{file_format}"'},
    )
