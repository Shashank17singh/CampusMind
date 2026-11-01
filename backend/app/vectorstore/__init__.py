"""Vector store, embeddings and hybrid retrieval. VECTOR_BACKEND selects the store."""

from functools import lru_cache

from app.config import get_settings
from app.db import get_session_factory
from app.vectorstore.base import ScoredChunk, SearchFilters, VectorStore
from app.vectorstore.bm25 import BM25Index
from app.vectorstore.embeddings import get_embedder
from app.vectorstore.hybrid import HybridRetriever, RetrievalResult, SearchMode
from app.vectorstore.rerank import get_reranker

__all__ = [
    "HybridRetriever",
    "RetrievalResult",
    "ScoredChunk",
    "SearchFilters",
    "SearchMode",
    "VectorStore",
    "get_retriever",
    "get_vector_store",
]


@lru_cache
def get_vector_store() -> VectorStore:
    """Return the configured store, importing only the backend that is in use."""
    settings = get_settings()
    if settings.vector_backend == "qdrant":
        from app.vectorstore.qdrant_store import QdrantVectorStore

        return QdrantVectorStore(settings.qdrant_url)

    from app.vectorstore.chroma_store import ChromaVectorStore

    return ChromaVectorStore(settings.chroma_path, settings.chroma_host, settings.chroma_port)


@lru_cache
def get_retriever() -> HybridRetriever:
    """Return the process-wide retriever."""
    settings = get_settings()
    return HybridRetriever(
        get_vector_store(),
        get_embedder(),
        BM25Index(get_session_factory()),
        reranker=get_reranker() if settings.rerank_enabled else None,
        vector_weight=settings.rrf_vector_weight,
        bm25_weight=settings.rrf_bm25_weight,
    )
