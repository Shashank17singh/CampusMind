"""The abstract vector store and the types it exchanges.

College-wide documents have no department or semester. Stores keep those as an empty
string and 0, and a filter on department "CSE" matches "CSE" or empty. A CSE student must
still see a circular addressed to everyone.
"""

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.ingestion.types import ChunkRecord

NO_DEPARTMENT = ""
NO_SEMESTER = 0


@dataclass(frozen=True)
class SearchFilters:
    """Metadata restrictions on a search."""

    department: str | None = None
    semester: int | None = None
    doc_type: str | None = None
    include_superseded: bool = False


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk and its vector."""

    record: ChunkRecord
    embedding: np.ndarray


@dataclass(frozen=True)
class ScoredChunk:
    """A search hit. `score` means whatever the search that produced it says it means."""

    chunk_id: uuid.UUID
    document_id: uuid.UUID
    title: str
    text: str
    score: float
    page_number: int | None
    section_heading: str | None
    is_table: bool
    department: str | None
    semester: int | None
    doc_type: str | None
    uploaded_at: str


def to_metadata(record: ChunkRecord) -> dict[str, Any]:
    """Flatten a chunk's metadata into the scalar values both stores accept."""
    return {
        "document_id": str(record.document_id),
        "title": record.title,
        "chunk_index": record.chunk_index,
        "page_number": record.page_number if record.page_number is not None else -1,
        "section_heading": record.section_heading or "",
        "is_table": record.is_table,
        "department": record.department or NO_DEPARTMENT,
        "semester": record.semester or NO_SEMESTER,
        "doc_type": record.doc_type or "",
        "uploaded_at": record.uploaded_at.isoformat(),
        "uploader_id": str(record.uploader_id) if record.uploader_id else "",
        "superseded": False,
    }


def from_metadata(chunk_id: str, text: str, score: float, meta: dict[str, Any]) -> ScoredChunk:
    """Rebuild a search hit from stored metadata."""
    page = int(meta["page_number"])
    return ScoredChunk(
        chunk_id=uuid.UUID(chunk_id),
        document_id=uuid.UUID(str(meta["document_id"])),
        title=str(meta["title"]),
        text=text,
        score=score,
        page_number=page if page >= 0 else None,
        section_heading=str(meta["section_heading"]) or None,
        is_table=bool(meta["is_table"]),
        department=str(meta["department"]) or None,
        semester=int(meta["semester"]) or None,
        doc_type=str(meta["doc_type"]) or None,
        uploaded_at=str(meta["uploaded_at"]),
    )


class VectorStore(ABC):
    """Stores chunk vectors. Everything above this class is independent of the backend."""

    @abstractmethod
    def add_chunks(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Insert chunks, replacing any with the same id."""

    @abstractmethod
    def delete_document(self, document_id: uuid.UUID) -> None:
        """Remove every chunk of a document."""

    @abstractmethod
    def search(
        self, query_embedding: np.ndarray, k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]:
        """Return the `k` nearest chunks, with cosine similarity as the score."""

    @abstractmethod
    def count(self) -> int:
        """Return the total number of stored chunks."""

    @abstractmethod
    def mark_superseded(self, document_id: uuid.UUID) -> None:
        """Hide a document's chunks from searches that exclude superseded documents."""
