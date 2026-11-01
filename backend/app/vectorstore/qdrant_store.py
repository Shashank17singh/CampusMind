"""Qdrant implementation of the vector store."""

import uuid
from collections.abc import Sequence

import numpy as np
from qdrant_client import QdrantClient, models

from app.vectorstore.base import (
    NO_DEPARTMENT,
    NO_SEMESTER,
    EmbeddedChunk,
    ScoredChunk,
    SearchFilters,
    VectorStore,
    from_metadata,
    to_metadata,
)

COLLECTION = "campusmind_chunks"
UPSERT_BATCH = 256


def _match(key: str, value: bool | int | str) -> models.FieldCondition:
    """Build an equality condition."""
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _match_any(key: str, values: list) -> models.FieldCondition:
    """Build a condition that accepts any of several values."""
    return models.FieldCondition(key=key, match=models.MatchAny(any=values))


def _build_filter(filters: SearchFilters | None) -> models.Filter | None:
    """Translate filters into a Qdrant filter."""
    active = filters or SearchFilters()
    conditions: list[models.Condition] = []
    if not active.include_superseded:
        conditions.append(_match("superseded", False))
    if active.department:
        conditions.append(_match_any("department", [active.department, NO_DEPARTMENT]))
    if active.semester:
        conditions.append(_match_any("semester", [active.semester, NO_SEMESTER]))
    if active.doc_type:
        conditions.append(_match("doc_type", active.doc_type))
    return models.Filter(must=conditions) if conditions else None


def _document_selector(document_id: uuid.UUID) -> models.FilterSelector:
    """Select every point belonging to a document."""
    return models.FilterSelector(
        filter=models.Filter(must=[_match("document_id", str(document_id))])
    )


class QdrantVectorStore(VectorStore):
    """Stores vectors in Qdrant, with cosine distance."""

    def __init__(self, url: str = ":memory:", dimension: int = 384) -> None:
        """Connect to a server URL, or use Qdrant's embedded mode for ":memory:"."""
        self._client = QdrantClient(location=url)
        if not self._client.collection_exists(COLLECTION):
            self._client.create_collection(
                COLLECTION,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )

    def add_chunks(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Upsert chunks in batches. The text is kept in the payload beside the metadata."""
        for start in range(0, len(chunks), UPSERT_BATCH):
            self._client.upsert(
                COLLECTION,
                points=[
                    models.PointStruct(
                        id=str(chunk.record.id),
                        vector=chunk.embedding.tolist(),
                        payload={**to_metadata(chunk.record), "text": chunk.record.text},
                    )
                    for chunk in chunks[start : start + UPSERT_BATCH]
                ],
            )

    def delete_document(self, document_id: uuid.UUID) -> None:
        """Remove every chunk of a document."""
        self._client.delete(COLLECTION, points_selector=_document_selector(document_id))

    def search(
        self, query_embedding: np.ndarray, k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]:
        """Return the nearest chunks. Qdrant reports cosine similarity directly."""
        result = self._client.query_points(
            COLLECTION,
            query=query_embedding.tolist(),
            query_filter=_build_filter(filters),
            limit=k,
            with_payload=True,
        )
        hits = []
        for point in result.points:
            payload = dict(point.payload or {})
            text = str(payload.pop("text", ""))
            hits.append(from_metadata(str(point.id), text, float(point.score), payload))
        return hits

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self._client.count(COLLECTION, exact=True).count

    def mark_superseded(self, document_id: uuid.UUID) -> None:
        """Flag every chunk of a document."""
        self._client.set_payload(
            COLLECTION, payload={"superseded": True}, points=_document_selector(document_id)
        )
