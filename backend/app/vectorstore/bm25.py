"""Keyword search with BM25 over the chunk text in PostgreSQL.

rank_bm25 keeps its index in memory, so each process builds it on first use and rebuilds
when a fingerprint of the corpus changes. That is fine for a college's thousands of
chunks. At millions, this would move to Postgres full-text search or Qdrant sparse vectors.
"""

import re
import threading
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi
from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from app.logging import get_logger
from app.models import ChunkMeta, Document
from app.vectorstore.base import ScoredChunk, SearchFilters

log = get_logger(__name__)

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase and split on anything that is not a letter or digit."""
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class _Row:
    """The fields of a chunk that a search needs."""

    chunk: ScoredChunk
    superseded: bool


class BM25Index:
    """An in-memory BM25 index of every chunk in a ready document."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Create an empty index. It is built on the first search."""
        self._session_factory = session_factory
        self._lock = threading.Lock()
        self._fingerprint: tuple[object, ...] | None = None
        self._rows: list[_Row] = []
        self._bm25: BM25Okapi | None = None

    def _current_fingerprint(self, session: Session) -> tuple[object, ...]:
        """Summarise the corpus. Any upload, delete or supersede changes this."""
        row = session.execute(
            select(
                func.count(ChunkMeta.id),
                func.max(ChunkMeta.created_at),
                func.max(Document.updated_at),
                func.count(Document.superseded_by_id),
            )
            .join(Document, ChunkMeta.document_id == Document.id)
            .where(Document.status == "ready")
        ).one()
        return tuple(row)

    def _rebuild(self, session: Session, fingerprint: tuple[object, ...]) -> None:
        """Load every ready chunk and build the index."""
        rows = session.execute(
            select(ChunkMeta, Document)
            .join(Document, ChunkMeta.document_id == Document.id)
            .where(Document.status == "ready")
            .order_by(ChunkMeta.document_id, ChunkMeta.chunk_index)
        ).all()
        self._rows = [
            _Row(
                ScoredChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    title=document.title,
                    text=chunk.text,
                    score=0.0,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    is_table=chunk.is_table,
                    department=document.department,
                    semester=document.semester,
                    doc_type=document.doc_type,
                    uploaded_at=document.uploaded_at.isoformat(),
                ),
                superseded=document.superseded_by_id is not None,
            )
            for chunk, document in rows
        ]
        corpus = [
            tokenize(f"{row.chunk.title} {row.chunk.section_heading or ''} {row.chunk.text}")
            for row in self._rows
        ]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._fingerprint = fingerprint
        log.info("bm25_index_built", chunks=len(corpus))

    def _refresh(self) -> None:
        """Rebuild if the corpus has changed since the last search."""
        with self._session_factory() as session:
            fingerprint = self._current_fingerprint(session)
            if fingerprint != self._fingerprint:
                self._rebuild(session, fingerprint)

    @staticmethod
    def _allowed(row: _Row, filters: SearchFilters) -> bool:
        """Apply the same filter meaning as the vector stores."""
        chunk = row.chunk
        if row.superseded and not filters.include_superseded:
            return False
        if filters.department and chunk.department not in (filters.department, None):
            return False
        if filters.semester and chunk.semester not in (filters.semester, None):
            return False
        return not (filters.doc_type and chunk.doc_type != filters.doc_type)

    def search(self, query: str, k: int, filters: SearchFilters | None = None) -> list[ScoredChunk]:
        """Return the top `k` chunks that contain at least one query term."""
        active = filters or SearchFilters()
        terms = tokenize(query)
        with self._lock:
            self._refresh()
            if self._bm25 is None or not terms:
                return []
            scores = np.asarray(self._bm25.get_scores(terms))
            rows = self._rows
            # BM25's idf goes negative for very common terms, so score alone cannot show
            # whether a chunk matched. A term must really occur for the chunk to count.
            frequencies = self._bm25.doc_freqs

        hits = []
        for index in np.argsort(-scores):
            row = rows[int(index)]
            if not self._allowed(row, active) or not any(t in frequencies[index] for t in terms):
                continue
            chunk = row.chunk
            hits.append(
                ScoredChunk(**{**chunk.__dict__, "score": float(scores[index])}),
            )
            if len(hits) == k:
                break
        return hits
