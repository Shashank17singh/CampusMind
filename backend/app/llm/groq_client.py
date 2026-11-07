"""Hosted LLM implementation using Groq's OpenAI-compatible chat API."""

import json
import time
from collections.abc import Callable, Iterator
from typing import Any

import httpx

from app.llm.base import ChatMessage, LLMClient, LLMError, LLMResponse, StreamResult
from app.logging import get_logger

log = get_logger(__name__)

BASE_URL = "https://api.groq.com/openai/v1"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class GroqClient(LLMClient):
    """Calls Groq's /chat/completions endpoint, retrying rate limits and server errors."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Create the client. Pass `transport` and `sleep` to inject test doubles."""
        self._model = model
        self._max_retries = max_retries
        self._sleep = sleep
        self._http = httpx.Client(
            base_url=BASE_URL,
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def _payload(
        self,
        messages: list[ChatMessage],
        system: str,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the request body shared by both call styles.

        JSON mode works on every Groq model, unlike strict schemas, so the schema goes in
        the system prompt and the reply is validated afterwards by `complete_json`.
        """
        if json_schema is not None:
            system = (
                f"{system}\n\nReply with one JSON object that matches this JSON schema, "
                f"and nothing else:\n{json.dumps(json_schema)}"
            ).strip()
        chat = [{"role": m.role, "content": m.content} for m in messages]
        if system:
            chat.insert(0, {"role": "system", "content": system})
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": chat,
            "max_completion_tokens": max_tokens,
        }
        if self._model.startswith("openai/gpt-oss"):
            # Reasoning tokens count against max_completion_tokens, so keep them few and
            # leave them out of the reply.
            payload["reasoning_effort"] = "low"
            payload["include_reasoning"] = False
        if json_schema is not None:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _wait(self, response: httpx.Response, attempt: int) -> float:
        """Seconds to wait before retrying: Groq's retry-after, else exponential backoff."""
        try:
            return float(response.headers["retry-after"])
        except (KeyError, ValueError):
            return float(2**attempt)

    @staticmethod
    def _error(response: httpx.Response) -> str:
        """The error message Groq returned, or the status line."""
        try:
            message = response.json()["error"]["message"]
        except (ValueError, KeyError, TypeError):
            message = response.reason_phrase
        return f"{response.status_code} {message}"

    def _send(self, payload: dict[str, Any], stream: bool) -> httpx.Response:
        """Send the request, retrying 429 and 5xx. The caller closes a streamed response."""
        for attempt in range(self._max_retries + 1):
            request = self._http.build_request("POST", "/chat/completions", json=payload)
            response = self._http.send(request, stream=stream)
            if response.status_code < 400:
                return response
            if stream:
                response.read()
            response.close()
            if response.status_code not in RETRY_STATUSES or attempt == self._max_retries:
                raise LLMError(f"Groq request failed: {self._error(response)}")
            wait = self._wait(response, attempt)
            log.warning("llm_retry", provider="groq", status=response.status_code, wait=wait)
            self._sleep(wait)
        raise LLMError("Groq request failed")  # pragma: no cover, the loop always returns

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str = "",
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Return a full reply."""
        payload = self._payload(messages, system, max_tokens, json_schema)
        try:
            data = self._send(payload, stream=False).json()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            log.warning("llm_request_failed", provider="groq", error=str(exc))
            raise LLMError(f"Groq request failed: {exc}") from exc

        usage = data.get("usage") or {}
        choices = data.get("choices") or [{}]
        return LLMResponse(
            text=choices[0].get("message", {}).get("content") or "",
            model=data.get("model", self._model),
            tokens_in=usage.get("prompt_tokens", 0),
            tokens_out=usage.get("completion_tokens", 0),
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        system: str = "",
        max_tokens: int = 2048,
        result: StreamResult | None = None,
    ) -> Iterator[str]:
        """Yield text fragments from Groq's server-sent events."""
        sink = result if result is not None else StreamResult()
        sink.model = self._model
        payload = self._payload(messages, system, max_tokens)
        payload["stream"] = True
        try:
            response = self._send(payload, stream=True)
            try:
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_text = line[len("data:") :].strip()
                    if data_text == "[DONE]":
                        break
                    event = json.loads(data_text)
                    sink.model = event.get("model", sink.model)
                    for choice in event.get("choices") or []:
                        text = (choice.get("delta") or {}).get("content")
                        if text:
                            sink.text_parts.append(text)
                            yield text
                    usage = event.get("usage") or (event.get("x_groq") or {}).get("usage")
                    if usage:
                        sink.tokens_in = usage.get("prompt_tokens", 0)
                        sink.tokens_out = usage.get("completion_tokens", 0)
            finally:
                response.close()
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            log.warning("llm_stream_failed", provider="groq", error=str(exc))
            raise LLMError(f"Groq stream failed: {exc}") from exc
