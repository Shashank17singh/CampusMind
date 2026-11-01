"""Background summary jobs and file export.

A file a student uploads to summarize is processed here and then deleted. It never enters
the document store or the vector store, because search results count as official sources.
"""

import io
import uuid
from html import escape
from pathlib import Path, PurePath

import pymupdf
from sqlalchemy.orm import Session, sessionmaker

from app.agents.summarizer import (
    Mode,
    SourceChunk,
    Summarizer,
    SummaryContent,
    SummaryError,
    load_chunks,
    render_text,
)
from app.celery_app import celery_app
from app.config import Settings, get_settings
from app.db import get_session_factory
from app.ingestion.chunking import chunk_blocks
from app.ingestion.clean import clean_blocks
from app.ingestion.errors import IngestionError
from app.ingestion.extract import extract
from app.ingestion.ocr import TesseractOcr
from app.ingestion.validation import EXTENSIONS
from app.llm import LLMClient, get_llm
from app.logging import get_logger
from app.models import Summary

log = get_logger(__name__)

GENERIC_FAILURE = "The summary failed because of an internal error. Please try again."


def chunks_from_file(path: Path, filename: str, settings: Settings) -> list[SourceChunk]:
    """Extract an uploaded file without storing it as a document."""
    kind = EXTENSIONS[PurePath(filename).suffix.lower()]
    ocr = TesseractOcr(settings.tesseract_cmd)
    extracted = extract(path, kind, ocr, settings.ocr_min_chars, settings.ocr_dpi)
    blocks = clean_blocks(extracted.blocks, settings.header_footer_threshold)
    chunks = chunk_blocks(blocks, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    return [SourceChunk(chunk.text, chunk.page_number, chunk.section_heading) for chunk in chunks]


def run_summary(
    session_factory: sessionmaker[Session],
    llm: LLMClient,
    settings: Settings,
    summary_id: uuid.UUID,
    upload_path: Path | None = None,
) -> bool:
    """Produce a stored summary. Failures are recorded on the row, for the user to see."""
    with session_factory() as session:
        summary = session.get_one(Summary, summary_id)
        document_id, filename = summary.document_id, summary.source_name
        mode: Mode = summary.mode  # type: ignore[assignment]

    try:
        if upload_path is not None:
            chunks = chunks_from_file(upload_path, filename, settings)
        else:
            with session_factory() as session:
                chunks = load_chunks(session, document_id)  # type: ignore[arg-type]
        result = Summarizer(llm).run(chunks, mode)
    except (SummaryError, IngestionError) as exc:
        _finish(session_factory, summary_id, error=str(exc))
        return False
    except Exception:
        log.exception("summary_crashed", summary_id=str(summary_id))
        _finish(session_factory, summary_id, error=GENERIC_FAILURE)
        return False
    finally:
        if upload_path is not None:
            upload_path.unlink(missing_ok=True)

    with session_factory() as session:
        summary = session.get_one(Summary, summary_id)
        summary.content = {
            **result.content.model_dump(),
            "verified": result.verified,
        }
        summary.dropped_count = result.dropped_count
        summary.tokens_in = result.tokens_in
        summary.tokens_out = result.tokens_out
        summary.duration_ms = result.duration_ms
        summary.status = "ready"
        session.commit()
    return True


def _finish(session_factory: sessionmaker[Session], summary_id: uuid.UUID, *, error: str) -> None:
    """Record a failure."""
    with session_factory() as session:
        summary = session.get_one(Summary, summary_id)
        summary.status = "failed"
        summary.error_message = error
        session.commit()


@celery_app.task(name="summaries.run", acks_late=True, soft_time_limit=900, time_limit=960)
def run_summary_task(summary_id: str, upload_path: str | None = None) -> bool:
    """Run one summary."""
    return run_summary(
        get_session_factory(),
        get_llm(),
        get_settings(),
        uuid.UUID(summary_id),
        Path(upload_path) if upload_path else None,
    )


def to_markdown(title: str, content: SummaryContent) -> str:
    """Render a summary as a Markdown file."""
    return f"# {title}\n\n{render_text(content)}\n"


def to_pdf(title: str, content: SummaryContent) -> bytes:
    """Render a summary as a PDF. Formulas appear as their LaTeX source."""
    parts = [f"<h1>{escape(title)}</h1>"]
    for group in content.groups:
        if group.heading:
            parts.append(f"<h2>{escape(group.heading)}</h2>")
        for item in group.items:
            pages = f" (p. {', '.join(map(str, item.pages))})" if item.pages else ""
            if item.answer:
                parts.append(
                    f"<p><b>Q:</b> {escape(item.text)}<br/>"
                    f"<b>A:</b> {escape(item.answer)}{pages}</p>"
                )
            else:
                parts.append(f"<p>&bull; {escape(item.text)}{pages}</p>")

    buffer = io.BytesIO()
    writer = pymupdf.DocumentWriter(buffer)
    story = pymupdf.Story(html="".join(parts))
    page = pymupdf.paper_rect("a4")
    margin = (50, 50, -50, -50)
    more = 1
    while more:
        device = writer.begin_page(page)
        more, _ = story.place(page + margin)
        story.draw(device)
        writer.end_page()
    writer.close()
    return buffer.getvalue()
