"""Recording each node's work in the agent_traces table."""

import json
import uuid
from dataclasses import asdict, is_dataclass
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.logging import get_logger
from app.models import AgentTrace

log = get_logger(__name__)

MAX_STRING_CHARS = 2000


def _default(value: Any) -> Any:
    """Convert values JSON cannot represent."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return str(value)


def _shorten(value: Any) -> Any:
    """Cap long strings, so whole documents do not end up in the trace table."""
    if isinstance(value, str):
        return value if len(value) <= MAX_STRING_CHARS else f"{value[:MAX_STRING_CHARS]}..."
    if isinstance(value, dict):
        return {key: _shorten(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shorten(item) for item in value]
    return value


def to_json(value: dict[str, Any]) -> dict[str, Any]:
    """Make a state dict safe to store as JSONB."""
    return _shorten(json.loads(json.dumps(value, default=_default)))


class TraceRecorder:
    """Writes trace rows. A tracing failure is logged, and never fails the user's request."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        """Keep the session factory."""
        self._session_factory = session_factory

    def record(
        self,
        trace_id: uuid.UUID,
        step: int,
        node: str,
        node_input: dict[str, Any],
        output: dict[str, Any],
        latency_ms: int,
        tokens_in: int = 0,
        tokens_out: int = 0,
        error: str | None = None,
    ) -> None:
        """Store one node's execution."""
        try:
            with self._session_factory() as session:
                session.add(
                    AgentTrace(
                        trace_id=trace_id,
                        step=step,
                        node=node,
                        input=to_json(node_input),
                        output=to_json(output),
                        latency_ms=latency_ms,
                        tokens_in=tokens_in,
                        tokens_out=tokens_out,
                        error=error,
                    )
                )
                session.commit()
        except Exception:
            log.exception("trace_write_failed", trace_id=str(trace_id), node=node)
