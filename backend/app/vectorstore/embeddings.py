"""The embedding service.

bge-small-en-v1.5 reads at most 512 tokens and silently drops anything longer, but chunks
default to 800. Long chunks are therefore embedded as overlapping windows, and the window
vectors are averaged and normalised again, so the end of a chunk still counts.
"""

import threading
from functools import lru_cache
from typing import Any, Protocol

import numpy as np

from app.config import get_settings
from app.logging import get_logger

log = get_logger(__name__)

# BGE retrieval models expect this on queries, and only on queries.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32
WINDOW_OVERLAP_TOKENS = 50


class Embedder(Protocol):
    """Turns text into normalised vectors."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages, returning one row per text."""
        ...

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a search query."""
        ...


def split_into_windows(tokens: list[int], size: int, overlap: int) -> list[list[int]]:
    """Cut a token sequence into windows of `size`, each sharing `overlap` with the last."""
    if overlap >= size:
        raise ValueError("Window overlap must be smaller than the window")
    if len(tokens) <= size:
        return [tokens]
    step = size - overlap
    windows = []
    for start in range(0, len(tokens), step):
        windows.append(tokens[start : start + size])
        if start + size >= len(tokens):
            break
    return windows


class SentenceTransformerEmbedder:
    """Wraps a sentence-transformers model, using the GPU when there is one."""

    def __init__(self, model_name: str, device: str = "", model: Any = None) -> None:
        """Load the model. Pass `model` to inject a test double."""
        if model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            log.info("loading_embedding_model", model=model_name, device=device)
            model = SentenceTransformer(model_name, device=device)
        self._model = model
        self.dimension: int = model.get_embedding_dimension()
        # The two special tokens take up two of the model's token slots.
        self._window_size: int = model.max_seq_length - 2
        self._window_overlap = min(WINDOW_OVERLAP_TOKENS, self._window_size // 4)

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Run the model."""
        return np.asarray(
            self._model.encode(
                texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Embed passages, averaging windows for any that are too long for the model."""
        tokenizer = self._model.tokenizer
        spans: list[tuple[int, int]] = []
        pieces: list[str] = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(tokens) <= self._window_size:
                windows = [text]
            else:
                windows = [
                    tokenizer.decode(window)
                    for window in split_into_windows(
                        tokens, self._window_size, self._window_overlap
                    )
                ]
            spans.append((len(pieces), len(pieces) + len(windows)))
            pieces.extend(windows)

        if not pieces:
            return np.empty((0, self.dimension), dtype=np.float32)

        vectors = self._encode(pieces)
        result = np.vstack([vectors[start:end].mean(axis=0) for start, end in spans])
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norms, 1e-12)

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a query with the prefix BGE models were trained with."""
        return self._encode([f"{QUERY_PREFIX}{text}"])[0]


_lock = threading.Lock()


@lru_cache
def _load_embedder() -> SentenceTransformerEmbedder:
    """Load the configured model."""
    settings = get_settings()
    return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_device)


def get_embedder() -> SentenceTransformerEmbedder:
    """Return the process-wide model. The lock stops two threads loading it at once."""
    with _lock:
        return _load_embedder()
