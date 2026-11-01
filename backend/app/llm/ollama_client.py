"""Local LLM implementation using the Ollama HTTP API."""

import json
from collections.abc import Iterator
from typing import Any

import httpx

from app.llm.base import ChatMessage, LLMClient, LLMError, LLMResponse, StreamResult
from app.logging import get_logger

log = get_logger(__name__)


class OllamaClient(LLMClient):
    """Calls a local Ollama server's /api/chat endpoint."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """Create the client. Pass `transport` to inject a test double."""
        self._model = model
        self._http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def _payload(
        self, messages: list[ChatMessage], system: str, max_tokens: int, stream: bool
    ) -> dict[str, Any]:
        """Build the request body shared by both call styles."""
        chat = [{"role": m.role, "content": m.content} for m in messages]
        if system:
            chat.insert(0, {"role": "system", "content": system})
        return {
            "model": self._model,
            "messages": chat,
            "stream": stream,
            "options": {"num_predict": max_tokens},
        }

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str = "",
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Return a full reply. Ollama accepts a JSON schema through `format`."""
        payload = self._payload(messages, system, max_tokens, stream=False)
        if json_schema is not None:
            payload["format"] = json_schema
        try:
            response = self._http.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            log.warning("llm_request_failed", provider="ollama", error=str(exc))
            raise LLMError(f"Ollama request failed: {exc}") from exc

        return LLMResponse(
            text=data.get("message", {}).get("content", ""),
            model=data.get("model", self._model),
            tokens_in=data.get("prompt_eval_count", 0),
            tokens_out=data.get("eval_count", 0),
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        system: str = "",
        max_tokens: int = 2048,
        result: StreamResult | None = None,
    ) -> Iterator[str]:
        """Yield text fragments from Ollama's newline-delimited JSON stream."""
        sink = result if result is not None else StreamResult()
        payload = self._payload(messages, system, max_tokens, stream=True)
        try:
            with self._http.stream("POST", "/api/chat", json=payload) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    data = json.loads(line)
                    text = data.get("message", {}).get("content", "")
                    if text:
                        sink.text_parts.append(text)
                        yield text
                    if data.get("done"):
                        sink.model = data.get("model", self._model)
                        sink.tokens_in = data.get("prompt_eval_count", 0)
                        sink.tokens_out = data.get("eval_count", 0)
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            log.warning("llm_stream_failed", provider="ollama", error=str(exc))
            raise LLMError(f"Ollama stream failed: {exc}") from exc
