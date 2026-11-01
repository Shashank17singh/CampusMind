"""ChromaDB implementation of the vector store."""

import uuid
from collections.abc import Sequence
from typing import Any

import chromadb
import numpy as np

from app.logging import get_logger
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

log = get_logger(__name__)

COLLECTION = "campusmind_chunks"


def _build_where(filters: SearchFilters | None) -> dict[str, Any] | None:
    """Translate filters into Chroma's `where` syntax."""
    active = filters or SearchFilters()
    clauses: list[dict[str, Any]] = []
    if not active.include_superseded:
        clauses.append({"superseded": False})
    if active.department:
        clauses.append({"department": {"$in": [active.department, NO_DEPARTMENT]}})
    if active.semester:
        clauses.append({"semester": {"$in": [active.semester, NO_SEMESTER]}})
    if active.doc_type:
        clauses.append({"doc_type": active.doc_type})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


class ChromaVectorStore(VectorStore):
    """Stores vectors in Chroma.

    A persistent client assumes one process owns the folder, which suits development.
    When an API and a Celery worker run together, give a host to use Chroma's server.
    """

    def __init__(self, path: str = "./chroma_data", host: str = "", port: int = 8000) -> None:
        """Connect to a Chroma server if `host` is set, otherwise open a local folder."""
        client = (
            chromadb.HttpClient(host=host, port=port) if host else chromadb.PersistentClient(path)
        )
        self._client = client
        self._collection = client.get_or_create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Upsert chunks, in batches no larger than Chroma accepts."""
        batch_size = self._client.get_max_batch_size()
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            self._collection.upsert(
                ids=[str(chunk.record.id) for chunk in batch],
                embeddings=np.stack([chunk.embedding for chunk in batch]),
                documents=[chunk.record.text for chunk in batch],
                metadatas=[to_metadata(chunk.record) for chunk in batch],
            )

    def delete_document(self, document_id: uuid.UUID) -> None:
        """Remove every chunk of a document."""
        self._collection.delete(where={"document_id": str(document_id)})

    def search(
        self, query_embedding: np.ndarray, k: int, filters: SearchFilters | None = None
    ) -> list[ScoredChunk]:
        """Return the nearest chunks. Chroma reports cosine distance, so 1 - d is similarity."""
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=_build_where(filters),
            include=["documents", "metadatas", "distances"],
        )
        return [
            from_metadata(chunk_id, text, 1.0 - float(distance), dict(meta))
            for chunk_id, text, distance, meta in zip(
                result["ids"][0],
                (result["documents"] or [[]])[0],
                (result["distances"] or [[]])[0],
                (result["metadatas"] or [[]])[0],
                strict=True,
            )
        ]

    def count(self) -> int:
        """Return the total number of stored chunks."""
        return self._collection.count()

    def mark_superseded(self, document_id: uuid.UUID) -> None:
        """Flag every chunk of a document."""
        ids = self._collection.get(where={"document_id": str(document_id)}, include=[])["ids"]
        if ids:
            self._collection.update(ids=ids, metadatas=[{"superseded": True}] * len(ids))
