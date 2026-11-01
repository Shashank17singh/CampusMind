"""Optional cross-encoder re-ranking."""

from dataclasses import replace
from functools import lru_cache
from typing import Protocol

from app.config import get_settings
from app.logging import get_logger
from app.vectorstore.base import ScoredChunk

log = get_logger(__name__)


class Reranker(Protocol):
    """Re-orders search hits by reading the query and each passage together."""

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Return the best `top_k` chunks, scored by the re-ranker."""
        ...


class CrossEncoderReranker:
    """Wraps a sentence-transformers cross-encoder such as bge-reranker-base."""

    def __init__(self, model_name: str, device: str = "") -> None:
        """Load the model."""
        import torch
        from sentence_transformers import CrossEncoder

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        log.info("loading_reranker", model=model_name, device=device)
        self._model = CrossEncoder(model_name, device=device)

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        """Score each (query, passage) pair and keep the best."""
        if not chunks:
            return []
        scores = self._model.predict([(query, chunk.text) for chunk in chunks])
        ranked = sorted(zip(scores, chunks, strict=True), key=lambda pair: -float(pair[0]))
        return [replace(chunk, score=float(score)) for score, chunk in ranked[:top_k]]


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    """Return the process-wide re-ranker, loading it on first use."""
    settings = get_settings()
    return CrossEncoderReranker(settings.reranker_model, settings.embedding_device)
