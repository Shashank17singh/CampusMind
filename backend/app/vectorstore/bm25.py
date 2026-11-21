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

_TOKEN = re.compile(r"[a-zA-Z0-9\u0900-\u0963\u0966-\u0D7F]+")

_EXPANSIONS: dict[str, list[str]] = {
    "cse": ["computer", "science", "engineering"],
    "ai": ["artificial", "intelligence", "cse-ai", "cseai"],
    "cseai": ["cse", "ai", "artificial", "intelligence", "computer", "science", "branch"],
    "cse-ai": ["cse", "ai", "artificial", "intelligence", "computer", "science", "branch"],
    "ml": ["machine", "learning"],
    "it": ["information", "technology"],
    "ece": ["electronics", "communication"],
    "ee": ["electrical"],
    "me": ["mechanical"],
    "mech": ["mechanical"],
    "ce": ["civil"],
    "sem": ["semester", "सेमेस्टर"],
    "sems": ["semester", "सेमेस्टर"],
    "btech": ["b", "tech"],
    "mtech": ["m", "tech"],
    "ba": ["bachelor", "arts", "बीए"],
    "ma": ["master", "arts", "एमए"],
    "bsc": ["bachelor", "science", "बीएससी"],
    "msc": ["master", "science", "एमएससी"],
    "bcom": ["bachelor", "commerce", "बीकॉम"],
    "mcom": ["master", "commerce", "एमकॉम"],
    "bba": ["bachelor", "business", "administration"],
    "mba": ["master", "business", "administration"],
    "llb": ["law", "laws", "एलएलबी", "विधि"],
    "llm": ["master", "laws", "law", "एलएलएम"],
    "bed": ["bachelor", "education", "बीएड"],
    "med": ["master", "education", "एमएड"],
    "bca": ["bachelor", "computer", "applications"],
    "mca": ["master", "computer", "applications"],
    "bpharm": ["bachelor", "pharmacy"],
    "bva": ["bachelor", "visual", "arts", "fine", "arts"],
    "mva": ["master", "visual", "arts"],
    "phd": ["doctorate", "research"],
    "arts": ["कला", "humanities", "social", "sciences"],
    "science": ["विज्ञान", "sciences"],
    "commerce": ["वाणिज्य", "business"],
    "law": ["विधि", "कानून", "legal"],
    "education": ["शिक्षा", "training"],
    "library": ["पुस्तकालय", "tagore", "books"],
    "hostel": ["छात्रावास", "हॉस्टल", "mess", "allotment"],
    "scholarship": ["छात्रवृत्ति", "financial", "aid"],
    "placement": ["placements", "jobs", "internships", "recruiters"],
    "admission": ["admissions", "प्रवेश", "दाखिला", "counselling"],
    "1st": ["first", "1", "i", "प्रथम", "पहला"],
    "2nd": ["second", "2", "ii", "द्वितीय", "दूसरा"],
    "3rd": ["third", "3", "iii", "तृतीय", "तीसरा"],
    "4th": ["fourth", "4", "iv", "चतुर्थ", "चौथा"],
    "5th": ["fifth", "5", "v", "पंचम", "पांचवां"],
    "6th": ["sixth", "6", "vi", "छठा"],
    "7th": ["seventh", "7", "vii", "सातवां"],
    "8th": ["eighth", "8", "viii", "आठवां"],
    "exam": ["exams", "examination", "परीक्षा", "परीक्षाएं"],
    "exams": ["exam", "examination", "परीक्षा", "परीक्षाएं"],
    "examination": ["exam", "exams", "परीक्षा", "परीक्षाएं"],
    "परीक्षा": ["exam", "exams", "examination"],
    "परीक्षाएं": ["exam", "exams", "examination"],
    "fee": ["fees", "शुल्क", "फीस"],
    "fees": ["fee", "शुल्क", "फीस"],
    "शुल्क": ["fee", "fees"],
    "फीस": ["fee", "fees"],
    "notice": ["notices", "circular", "सूचना", "परिपत्र", "नोटिस"],
    "circular": ["circulars", "notice", "सूचना", "परिपत्र"],
    "सूचना": ["notice", "circular"],
    "परिपत्र": ["notice", "circular"],
    "syllabus": ["पाठ्यक्रम", "सिलेबस"],
    "पाठ्यक्रम": ["syllabus"],
    "attendance": ["उपस्थिति", "हाजिरी"],
    "उपस्थिति": ["attendance"],
    "result": ["results", "परिणाम", "रिजल्ट"],
    "परिणाम": ["result", "results"],
    "semester": ["sem", "सेमेस्टर"],
    "सेमेस्टर": ["semester", "sem"],
    "प्रवेश": ["admission", "admissions"],
    "छात्रावास": ["hostel"],
    "holiday": ["holidays", "अवकाश", "छुट्टी"],
    "छुट्टी": ["holiday", "holidays", "अवकाश"],
    "अवकाश": ["holiday", "holidays"],
}


def tokenize(text: str) -> list[str]:
    """Lowercase and split on anything that is not a letter or digit, expanding common abbreviations."""
    raw_tokens = _TOKEN.findall(text.lower())
    result = list(raw_tokens)
    for token in raw_tokens:
        if token in _EXPANSIONS:
            result.extend(_EXPANSIONS[token])
    return result


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
        if filters.department and chunk.department not in (*filters.departments(), None):
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
