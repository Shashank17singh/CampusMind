"""Upload handling: validation, exact-duplicate detection and document creation."""

import hashlib
import uuid
from dataclasses import dataclass
from pathlib import PurePath

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ingestion.storage import LocalStorage
from app.ingestion.validation import validate_upload
from app.logging import get_logger
from app.models import Document

log = get_logger(__name__)


@dataclass
class UploadResult:
    """The stored document, and whether the pipeline needs to run for it."""

    document: Document
    duplicate: bool


def _prepare_retry(document: Document) -> None:
    """Reset a failed document so uploading the same file again retries it."""
    document.status = "uploaded"
    document.progress = 0
    document.error_message = None


def create_document(
    session: Session,
    storage: LocalStorage,
    *,
    filename: str,
    data: bytes,
    uploader_id: uuid.UUID | None,
    title: str | None = None,
    doc_type: str | None = None,
    department: str | None = None,
    semester: int | None = None,
) -> UploadResult:
    """Validate an upload and create its document, or return the existing identical one."""
    _, mime_type = validate_upload(filename, data, get_settings().max_upload_bytes)
    sha256 = hashlib.sha256(data).hexdigest()

    existing = session.scalar(select(Document).where(Document.sha256 == sha256))
    if existing is not None:
        if existing.status == "failed":
            _prepare_retry(existing)
            return UploadResult(existing, duplicate=False)
        log.info("duplicate_upload_skipped", document_id=str(existing.id), sha256=sha256)
        return UploadResult(existing, duplicate=True)

    path = PurePath(filename)
    document = Document(
        title=title or path.stem.replace("_", " ").strip() or path.name,
        filename=path.name,
        mime_type=mime_type,
        size_bytes=len(data),
        sha256=sha256,
        storage_path=storage.save(sha256, path.suffix, data),
        doc_type=doc_type,
        department=department,
        semester=semester,
        uploader_id=uploader_id,
    )
    try:
        with session.begin_nested():
            session.add(document)
    except IntegrityError:
        # Two identical uploads raced, and the other one reached the unique index first.
        winner = session.scalar(select(Document).where(Document.sha256 == sha256))
        if winner is None:
            raise
        return UploadResult(winner, duplicate=True)
    return UploadResult(document, duplicate=False)
