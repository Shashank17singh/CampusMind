"""Test doubles shared across the suite."""

import hashlib
import re
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
from PIL import Image

from app.ingestion.types import ChunkRecord
from app.llm import ChatMessage, LLMClient, LLMError, LLMResponse, StreamResult


class FakeLLM(LLMClient):
    """Returns scripted replies in order, and records every call."""

    def __init__(self, replies: Sequence[str | Exception] = ()) -> None:
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def _next(self) -> str:
        if not self.replies:
            raise LLMError("FakeLLM has no reply left")
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str = "",
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "system": system, "schema": json_schema})
        return LLMResponse(text=self._next(), model="fake", tokens_in=10, tokens_out=5)

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        system: str = "",
        max_tokens: int = 2048,
        result: StreamResult | None = None,
    ) -> Iterator[str]:
        self.calls.append({"messages": messages, "system": system, "schema": None})
        sink = result if result is not None else StreamResult()
        for word in self._next().split(" "):
            sink.text_parts.append(f"{word} ")
            yield f"{word} "
        sink.model, sink.tokens_in, sink.tokens_out = "fake", 10, 5


class RoutedLLM(LLMClient):
    """Answers from a function of the prompt, for calls that run in parallel."""

    def __init__(self, respond: Callable[[str, str], str]) -> None:
        self.respond = respond
        self.active = 0
        self.peak = 0
        self.calls = 0
        self._lock = threading.Lock()

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str = "",
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        with self._lock:
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            time.sleep(0.01)
            text = self.respond(system, messages[-1].content)
        finally:
            with self._lock:
                self.active -= 1
        return LLMResponse(text=text, model="fake", tokens_in=100, tokens_out=20)

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        system: str = "",
        max_tokens: int = 2048,
        result: StreamResult | None = None,
    ) -> Iterator[str]:
        raise NotImplementedError


class FakeOcr:
    """Returns fixed text, and counts how often it was called."""

    def __init__(self, text: str = "") -> None:
        self.text = text
        self.calls = 0

    def image_to_text(self, image: Image.Image) -> str:
        self.calls += 1
        return self.text


class HashingEmbedder:
    """A deterministic bag-of-words embedder, so tests need no model download."""

    def __init__(self, dimension: int = 64) -> None:
        self.dimension = dimension

    def _embed(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimension, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            bucket = int(hashlib.md5(word.encode(), usedforsecurity=False).hexdigest(), 16)
            vector[bucket % self.dimension] += 1.0
        norm = np.linalg.norm(vector)
        if norm == 0:
            vector[0] = 1.0
            return vector
        return vector / norm

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed(text) for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed(text)


class FakeIndexer:
    """Stores records in memory."""

    def __init__(self, fail_with: Exception | None = None) -> None:
        self.records: dict[uuid.UUID, list[ChunkRecord]] = {}
        self.superseded: set[uuid.UUID] = set()
        self.fail_with = fail_with

    def index(
        self, records: Sequence[ChunkRecord], on_progress: Callable[[int, int], None]
    ) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        for record in records:
            self.records.setdefault(record.document_id, []).append(record)
        on_progress(len(records), len(records))

    def delete(self, document_id: uuid.UUID) -> None:
        self.records.pop(document_id, None)

    def mark_superseded(self, document_id: uuid.UUID) -> None:
        self.superseded.add(document_id)
