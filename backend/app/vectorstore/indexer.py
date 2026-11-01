"""Connects the ingestion pipeline to the embedder and vector store."""

import uuid
from collections.abc import Callable, Sequence

from app.ingestion.types import ChunkRecord
from app.vectorstore.base import EmbeddedChunk, VectorStore
from app.vectorstore.embeddings import BATCH_SIZE, Embedder


def embedding_text(record: ChunkRecord) -> str:
    """Return the text to embed. Title and heading give a chunk its missing context."""
    parts = [record.title, record.section_heading, record.text]
    return "\n".join(part for part in parts if part)


class VectorIndexer:
    """Embeds chunks in batches and writes them to the store."""

    def __init__(self, store: VectorStore, embedder: Embedder) -> None:
        """Keep the store and embedder."""
        self._store = store
        self._embedder = embedder

    def index(
        self, records: Sequence[ChunkRecord], on_progress: Callable[[int, int], None]
    ) -> None:
        """Embed and store records, reporting progress after each batch."""
        for start in range(0, len(records), BATCH_SIZE):
            batch = records[start : start + BATCH_SIZE]
            vectors = self._embedder.embed_documents([embedding_text(r) for r in batch])
            self._store.add_chunks(
                [EmbeddedChunk(r, vector) for r, vector in zip(batch, vectors, strict=True)]
            )
            on_progress(min(start + BATCH_SIZE, len(records)), len(records))

    def delete(self, document_id: uuid.UUID) -> None:
        """Remove a document's vectors."""
        self._store.delete_document(document_id)

    def mark_superseded(self, document_id: uuid.UUID) -> None:
        """Hide a document's vectors from default searches."""
        self._store.mark_superseded(document_id)
