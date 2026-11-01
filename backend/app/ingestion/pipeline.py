"""The ingestion pipeline: uploaded -> extracting -> chunking -> embedding -> ready | failed."""

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta
from pathlib import PurePath

from sqlalchemy import delete, or_, update
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.ingestion.chunking import chunk_blocks
from app.ingestion.clean import clean_blocks
from app.ingestion.dedupe import find_superseded
from app.ingestion.errors import ExtractionError, IngestionError
from app.ingestion.extract import extract
from app.ingestion.storage import LocalStorage
from app.ingestion.tagging import TagResult, auto_tag
from app.ingestion.types import Chunk, ChunkRecord, Indexer, OcrEngine
from app.ingestion.validation import EXTENSIONS
from app.llm import LLMClient
from app.logging import get_logger
from app.models import ChunkMeta, Document
from app.models.base import utcnow

log = get_logger(__name__)

STALE_CLAIM = timedelta(minutes=15)
PROGRESS_EXTRACTING = 5
PROGRESS_CHUNKING = 40
PROGRESS_EMBEDDING = 60
PROGRESS_EMBEDDED = 95
GENERIC_FAILURE = "Ingestion failed because of an internal error. Please try again."


@dataclass
class PipelineDeps:
    """Everything the pipeline needs, passed in so tests can replace any part."""

    session_factory: sessionmaker[Session]
    storage: LocalStorage
    ocr: OcrEngine
    llm: LLMClient
    indexer: Indexer
    settings: Settings
    on_ready: Callable[[uuid.UUID], None] | None = None


def _set_status(
    deps: PipelineDeps,
    document_id: uuid.UUID,
    status: str,
    progress: int,
    error_message: str | None = None,
) -> None:
    """Commit a status change straight away so pollers can see it."""
    with deps.session_factory() as session:
        session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                status=status,
                progress=progress,
                error_message=error_message,
                updated_at=utcnow(),
            )
        )
        session.commit()
    log.info("ingestion_status", document_id=str(document_id), status=status, progress=progress)


def _claim(deps: PipelineDeps, document_id: uuid.UUID) -> bool:
    """Atomically take the document, so a redelivered task cannot run it twice.

    A run that died mid-way leaves a non-terminal status behind. Those are claimable again
    once they have gone quiet for longer than a task's time limit.
    """
    with deps.session_factory() as session:
        claimed = session.scalar(
            update(Document)
            .where(
                Document.id == document_id,
                or_(
                    Document.status.in_(("uploaded", "failed")),
                    Document.updated_at < utcnow() - STALE_CLAIM,
                ),
                Document.status != "ready",
            )
            .values(
                status="extracting",
                progress=PROGRESS_EXTRACTING,
                error_message=None,
                updated_at=utcnow(),
            )
            .returning(Document.id)
        )
        session.commit()
    return claimed is not None


def _save_chunks(
    deps: PipelineDeps, document_id: uuid.UUID, chunks: list[Chunk], tags: TagResult, pages: int
) -> list[ChunkRecord]:
    """Replace the document's chunks and tags, and build the records to index."""
    with deps.session_factory() as session:
        document = session.get_one(Document, document_id)
        document.doc_type = tags.doc_type
        document.department = tags.department
        document.semester = tags.semester
        document.academic_year = tags.academic_year
        document.page_count = pages

        session.execute(delete(ChunkMeta).where(ChunkMeta.document_id == document_id))
        records = []
        for chunk in chunks:
            chunk_id = uuid.uuid4()
            session.add(
                ChunkMeta(
                    id=chunk_id,
                    document_id=document_id,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    is_table=chunk.is_table,
                )
            )
            records.append(
                ChunkRecord(
                    id=chunk_id,
                    document_id=document_id,
                    title=document.title,
                    chunk_index=chunk.chunk_index,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    is_table=chunk.is_table,
                    department=tags.department,
                    semester=tags.semester,
                    doc_type=tags.doc_type,
                    uploaded_at=document.uploaded_at,
                    uploader_id=document.uploader_id,
                )
            )
        session.commit()
    return records


def _embed(deps: PipelineDeps, document_id: uuid.UUID, records: list[ChunkRecord]) -> None:
    """Index the records, moving the progress bar through the embedding range."""
    span = PROGRESS_EMBEDDED - PROGRESS_EMBEDDING

    def report(done: int, total: int) -> None:
        progress = PROGRESS_EMBEDDING + span * done // max(total, 1)
        _set_status(deps, document_id, "embedding", progress)

    deps.indexer.delete(document_id)
    deps.indexer.index(records, report)


def _supersede_older_versions(deps: PipelineDeps, document_id: uuid.UUID, text: str) -> None:
    """Mark older versions of this circular as superseded, now the new one is searchable."""
    with deps.session_factory() as session:
        document = session.get_one(Document, document_id)
        older = find_superseded(session, document, text, deps.settings.near_duplicate_threshold)
        if older:
            session.execute(
                update(Document)
                .where(Document.id.in_(older))
                .values(superseded_by_id=document_id, updated_at=utcnow())
            )
        session.commit()
    for older_id in older:
        deps.indexer.mark_superseded(older_id)
        log.info("document_superseded", old=str(older_id), new=str(document_id))


def _discard_partial_output(deps: PipelineDeps, document_id: uuid.UUID) -> None:
    """Remove chunks and vectors from a failed run. Never hides the original error."""
    try:
        with deps.session_factory() as session:
            session.execute(delete(ChunkMeta).where(ChunkMeta.document_id == document_id))
            session.commit()
        deps.indexer.delete(document_id)
    except Exception:
        log.exception("ingestion_cleanup_failed", document_id=str(document_id))


def _run_stages(deps: PipelineDeps, document_id: uuid.UUID) -> None:
    """Run every stage in order."""
    with deps.session_factory() as session:
        document = session.get_one(Document, document_id)
        filename = document.filename
        path = deps.storage.path(document.storage_path)
        provided = TagResult.model_validate(
            {
                "department": document.department,
                "semester": document.semester,
                "doc_type": document.doc_type,
                "academic_year": document.academic_year,
            }
        )

    kind = EXTENSIONS[PurePath(filename).suffix.lower()]
    settings = deps.settings
    extracted = extract(path, kind, deps.ocr, settings.ocr_min_chars, settings.ocr_dpi)
    blocks = clean_blocks(extracted.blocks, settings.header_footer_threshold)
    full_text = "\n\n".join(block.text for block in blocks)

    _set_status(deps, document_id, "chunking", PROGRESS_CHUNKING)
    tags = auto_tag(filename, full_text, deps.llm, provided)
    chunks = chunk_blocks(blocks, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    if not chunks:
        raise ExtractionError("No text could be extracted. The file has headings but no content.")
    records = _save_chunks(deps, document_id, chunks, tags, extracted.page_count)

    _set_status(deps, document_id, "embedding", PROGRESS_EMBEDDING)
    _embed(deps, document_id, records)
    _supersede_older_versions(deps, document_id, full_text)


def run_ingestion(deps: PipelineDeps, document_id: uuid.UUID, *, force: bool = False) -> bool:
    """Process one document. Returns True if it reached `ready`.

    A failure is recorded on the document rather than raised, so the uploader sees the
    reason. Database outages are raised instead, because Celery can usefully retry them.
    `force` skips the claim, for those retries.
    """
    if not force and not _claim(deps, document_id):
        log.info("ingestion_skipped_not_claimable", document_id=str(document_id))
        return False

    try:
        _run_stages(deps, document_id)
    except IngestionError as exc:
        _discard_partial_output(deps, document_id)
        _set_status(deps, document_id, "failed", 0, str(exc))
        return False
    except OperationalError:
        raise
    except Exception:
        log.exception("ingestion_crashed", document_id=str(document_id))
        _discard_partial_output(deps, document_id)
        _set_status(deps, document_id, "failed", 0, GENERIC_FAILURE)
        return False

    _set_status(deps, document_id, "ready", 100)
    if deps.on_ready is not None:
        deps.on_ready(document_id)
    return True
