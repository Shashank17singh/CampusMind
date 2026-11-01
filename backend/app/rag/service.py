"""Builds the pipeline and renders it as Server-Sent Events."""

import json
import time
from collections.abc import Iterator
from functools import lru_cache
from typing import Any

from app.config import get_settings
from app.llm import LLMError, get_llm
from app.logging import get_logger
from app.rag.cache import AnswerCache
from app.rag.pipeline import RagPipeline
from app.rag.schemas import FinalEvent, RagRequest, TokenEvent
from app.redis_client import get_redis
from app.vectorstore import get_retriever

log = get_logger(__name__)


def get_answer_cache() -> AnswerCache:
    """Return the answer cache."""
    return AnswerCache(get_redis(), get_settings().rag_cache_ttl_seconds)


@lru_cache
def get_rag_pipeline() -> RagPipeline:
    """Return the process-wide pipeline."""
    return RagPipeline(get_llm(), get_retriever(), get_answer_cache(), get_settings())


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Format one Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"


def sse_stream(pipeline: RagPipeline, request: RagRequest, trace_id: str) -> Iterator[str]:
    """Render a pipeline run as `token`, `citations`, `done` and `error` events."""
    started = time.perf_counter()
    try:
        for event in pipeline.stream(request):
            if isinstance(event, TokenEvent):
                yield format_sse("token", {"text": event.text})
            elif isinstance(event, FinalEvent):
                final = event.answer
                yield format_sse(
                    "citations",
                    {
                        "citations": [c.model_dump(mode="json") for c in final.citations],
                        "confidence": final.confidence,
                        "suggested_followups": final.suggested_followups,
                    },
                )
                yield format_sse(
                    "done",
                    {
                        "latency_ms": int((time.perf_counter() - started) * 1000),
                        "tokens_in": final.tokens_in,
                        "tokens_out": final.tokens_out,
                    },
                )
    except LLMError as exc:
        log.error("rag_stream_failed", trace_id=trace_id, error=str(exc))
        yield format_sse(
            "error",
            {
                "code": "llm_unavailable",
                "message": "The assistant is unavailable right now. Please try again.",
                "trace_id": trace_id,
            },
        )
