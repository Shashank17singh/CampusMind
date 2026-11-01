"""Answer cache in Redis.

Keys include a generation number, and any document change bumps it, which makes every
older entry unreachable until it expires. Tracking which answers depend on which documents
would miss the case that matters most: a brand new circular cannot be "related" to an
answer written before it existed, yet it may change that answer.
"""

import hashlib
import json
import re

import redis

from app.logging import get_logger
from app.rag.schemas import RagAnswer
from app.vectorstore import SearchFilters

log = get_logger(__name__)

GENERATION_KEY = "rag:generation"


def normalize_question(question: str) -> str:
    """Lowercase, and strip punctuation and extra spaces, so trivial variants share a key."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", question.lower())).strip()


class AnswerCache:
    """Caches finished answers. Redis failures are logged and treated as a cache miss."""

    def __init__(self, client: redis.Redis, ttl_seconds: int = 3600) -> None:
        """Keep the client and expiry time."""
        self._client = client
        self._ttl = ttl_seconds

    def _key(self, question: str, filters: SearchFilters) -> str:
        """Build the key from the question, the filter set and the current generation."""
        generation = str(self._client.get(GENERATION_KEY) or "0")
        material = json.dumps(
            [normalize_question(question), filters.department, filters.semester, filters.doc_type]
        )
        digest = hashlib.sha256(material.encode()).hexdigest()
        return f"rag:answer:{generation}:{digest}"

    def get(self, question: str, filters: SearchFilters) -> RagAnswer | None:
        """Return a cached answer, if there is one."""
        try:
            raw = self._client.get(self._key(question, filters))
        except redis.RedisError as exc:
            log.warning("answer_cache_unavailable", error=str(exc))
            return None
        if raw is None:
            return None
        return RagAnswer.model_validate_json(raw).model_copy(update={"cached": True})

    def set(self, question: str, filters: SearchFilters, answer: RagAnswer) -> None:
        """Store an answer."""
        try:
            self._client.set(self._key(question, filters), answer.model_dump_json(), ex=self._ttl)
        except redis.RedisError as exc:
            log.warning("answer_cache_unavailable", error=str(exc))

    def invalidate(self) -> None:
        """Make all cached answers unreachable. Called whenever a document changes."""
        try:
            self._client.incr(GENERATION_KEY)
        except redis.RedisError as exc:
            log.error("answer_cache_invalidation_failed", error=str(exc))
