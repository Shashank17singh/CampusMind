"""Hybrid retrieval: vector search and BM25 merged with Reciprocal Rank Fusion."""

import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from app.vectorstore.base import ScoredChunk, SearchFilters, VectorStore
from app.vectorstore.bm25 import BM25Index
from app.vectorstore.embeddings import Embedder
from app.vectorstore.rerank import Reranker

RRF_K = 60
CANDIDATES = 20


class SearchMode(StrEnum):
    """The retrieval strategies compared by eval/retrieval_bench.py."""

    VECTOR = "vector"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANK = "hybrid_rerank"


def rrf_merge(
    rankings: Sequence[Sequence[uuid.UUID]],
    weights: Sequence[float] | None = None,
    k: int = RRF_K,
) -> list[tuple[uuid.UUID, float]]:
    """Merge ranked lists. Each item scores the sum of weight / (k + rank) over the lists.

    Using ranks, not raw scores, means cosine similarities and BM25 scores never have to
    be put on a common scale. Ties keep the order items were first seen in.
    """
    active_weights = weights if weights is not None else [1.0] * len(rankings)
    if len(active_weights) != len(rankings):
        raise ValueError("Pass exactly one weight per ranking")

    scores: dict[uuid.UUID, float] = {}
    for ranking, weight in zip(rankings, active_weights, strict=True):
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: -pair[1])


@dataclass
class RetrievalResult:
    """Search hits, plus the cosine similarity of each one found by vector search.

    RRF scores only say how chunks rank against each other. Deciding whether a result is
    good enough to answer from needs similarity, so the RAG layer reads this field.
    """

    chunks: list[ScoredChunk]
    similarities: dict[uuid.UUID, float]


class HybridRetriever:
    """Runs any of the search modes."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        bm25: BM25Index,
        reranker: Reranker | None = None,
        vector_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> None:
        """Wire up the search components. The re-ranker is optional."""
        self._store = store
        self._embedder = embedder
        self._bm25 = bm25
        self._reranker = reranker
        self._weights = (vector_weight, bm25_weight)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        filters: SearchFilters | None = None,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> RetrievalResult:
        """Return the top `k` chunks for a query."""
        if mode is SearchMode.BM25:
            return RetrievalResult(self._bm25.search(query, k, filters), {})

        vector_hits = self._store.search(self._embedder.embed_query(query), CANDIDATES, filters)
        similarities = {chunk.chunk_id: chunk.score for chunk in vector_hits}
        if mode is SearchMode.VECTOR:
            return RetrievalResult(vector_hits[:k], similarities)

        keyword_hits = self._bm25.search(query, CANDIDATES, filters)
        by_id = {chunk.chunk_id: chunk for chunk in [*keyword_hits, *vector_hits]}
        merged = rrf_merge(
            [
                [chunk.chunk_id for chunk in vector_hits],
                [chunk.chunk_id for chunk in keyword_hits],
            ],
            self._weights,
        )
        fused = [replace(by_id[chunk_id], score=score) for chunk_id, score in merged]

        if mode is SearchMode.HYBRID_RERANK:
            if self._reranker is None:
                raise ValueError("hybrid_rerank needs a re-ranker")
            return RetrievalResult(
                self._reranker.rerank(query, fused[:CANDIDATES], k), similarities
            )
        return RetrievalResult(fused[:k], similarities)
