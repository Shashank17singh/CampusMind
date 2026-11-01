"""Near-duplicate detection for new versions of circulars.

Exact duplicates are caught earlier, by the SHA-256 unique index on `documents`.
"""

import re
import uuid

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import ChunkMeta, Document

_SHINGLE_WORDS = 5
_MAX_CANDIDATES = 200


def shingles(text: str, size: int = _SHINGLE_WORDS) -> set[tuple[str, ...]]:
    """Return the set of overlapping word sequences in `text`."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < size:
        return {tuple(words)} if words else set()
    return {tuple(words[index : index + size]) for index in range(len(words) - size + 1)}


def jaccard(first: set[tuple[str, ...]], second: set[tuple[str, ...]]) -> float:
    """Return the Jaccard similarity of two sets."""
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)


def find_superseded(
    session: Session, document: Document, text: str, threshold: float
) -> list[uuid.UUID]:
    """Return older current circulars that this document is a new version of.

    Comparing against recent circulars in the same department keeps this cheap, and
    word shingles survive small edits such as a changed date or an added paragraph.
    """
    if document.doc_type != "circular":
        return []

    candidate_ids = session.scalars(
        select(Document.id)
        .where(
            Document.doc_type == "circular",
            Document.status == "ready",
            Document.superseded_by_id.is_(None),
            Document.id != document.id,
            Document.department.is_not_distinct_from(document.department),
            Document.uploaded_at <= document.uploaded_at,
        )
        .order_by(Document.uploaded_at.desc())
        .limit(_MAX_CANDIDATES)
    ).all()
    if not candidate_ids:
        return []

    rows = session.execute(
        select(ChunkMeta.document_id, ChunkMeta.text)
        .where(ChunkMeta.document_id.in_(candidate_ids))
        .order_by(ChunkMeta.document_id, ChunkMeta.chunk_index)
    ).all()
    texts: dict[uuid.UUID, list[str]] = {}
    for document_id, chunk_text in rows:
        texts.setdefault(document_id, []).append(chunk_text)

    new_shingles = shingles(text)
    return [
        document_id
        for document_id, parts in texts.items()
        if jaccard(new_shingles, shingles(" ".join(parts))) >= threshold
    ]
