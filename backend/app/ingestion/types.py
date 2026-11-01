"""Data passed between ingestion stages."""

import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol

from PIL import Image


@dataclass
class Block:
    """A paragraph, heading or table in reading order."""

    text: str
    page: int | None = None
    heading_level: int | None = None
    is_table: bool = False


@dataclass
class ExtractedDocument:
    """Everything the extractors got from a file."""

    blocks: list[Block]
    page_count: int
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    """A piece of a document ready for embedding."""

    text: str
    token_count: int
    chunk_index: int
    page_number: int | None
    section_heading: str | None
    is_table: bool = False


@dataclass(frozen=True)
class ChunkRecord:
    """A chunk plus the document metadata it needs for filtering and citation."""

    id: uuid.UUID
    document_id: uuid.UUID
    title: str
    chunk_index: int
    text: str
    token_count: int
    page_number: int | None
    section_heading: str | None
    is_table: bool
    department: str | None
    semester: int | None
    doc_type: str | None
    uploaded_at: datetime
    uploader_id: uuid.UUID | None


class OcrEngine(Protocol):
    """Turns an image into text."""

    def image_to_text(self, image: Image.Image) -> str:
        """Return the text found in `image`."""
        ...


class Indexer(Protocol):
    """Where finished chunks go. Implemented by the vector store layer."""

    def index(
        self, records: Sequence[ChunkRecord], on_progress: Callable[[int, int], None]
    ) -> None:
        """Embed and store records, reporting (done, total) as batches finish."""
        ...

    def delete(self, document_id: uuid.UUID) -> None:
        """Remove every vector belonging to a document."""
        ...

    def mark_superseded(self, document_id: uuid.UUID) -> None:
        """Flag a document's vectors so default searches skip them."""
        ...
