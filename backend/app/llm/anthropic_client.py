"""Hosted LLM implementation using the Anthropic SDK."""

from collections.abc import Iterator
from typing import Any

import anthropic

from app.llm.base import ChatMessage, LLMClient, LLMError, LLMResponse, StreamResult
from app.logging import get_logger

log = get_logger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# Models that accept a server-side fallback when a safety classifier declines a request.
# A chemistry lab manual or a biology syllabus can trip one, and a retry on another model
# is better than telling a student the assistant failed.
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")


def _strict_schema(schema: Any) -> Any:
    """Add `additionalProperties: false` to every object, as structured outputs require."""
    if isinstance(schema, dict):
        cleaned = {key: _strict_schema(value) for key, value in schema.items() if key != "title"}
        if cleaned.get("type") == "object":
            cleaned["additionalProperties"] = False
            cleaned["required"] = sorted(cleaned.get("properties", {}))
        return cleaned
    if isinstance(schema, list):
        return [_strict_schema(item) for item in schema]
    return schema


class AnthropicClient(LLMClient):
    """Calls the Claude Messages API."""

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 120.0,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        """Create the client. Pass `client` to inject a test double."""
        self._model = model
        self._client = client or anthropic.Anthropic(api_key=api_key or None, timeout=timeout)

    def _request(self, messages: list[ChatMessage], system: str, max_tokens: int) -> dict[str, Any]:
        """Build the arguments shared by both call styles."""
        request: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }
        if system:
            request["system"] = system
        if self._model in _FALLBACK_MODELS:
            request["betas"] = [FALLBACK_BETA]
            request["fallbacks"] = "default"
        return request

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str = "",
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Return a full reply, constrained to `json_schema` when one is given."""
        request = self._request(messages, system, max_tokens)
        if json_schema is not None:
            request["output_config"] = {
                "format": {"type": "json_schema", "schema": _strict_schema(json_schema)}
            }
        try:
            response = self._client.beta.messages.create(**request)
        except anthropic.APIError as exc:
            log.warning("llm_request_failed", provider="anthropic", error=str(exc))
            raise LLMError(f"Anthropic request failed: {exc}") from exc

        if response.stop_reason == "refusal":
            raise LLMError("Anthropic declined the request")
        text = "".join(block.text for block in response.content if block.type == "text")
        return LLMResponse(
            text=text,
            model=response.model,
            tokens_in=response.usage.input_tokens,
            tokens_out=response.usage.output_tokens,
        )

    def stream(
        self,
        messages: list[ChatMessage],
        *,
        system: str = "",
        max_tokens: int = 2048,
        result: StreamResult | None = None,
    ) -> Iterator[str]:
        """Yield text fragments as Claude produces them."""
        sink = result if result is not None else StreamResult()
        try:
            with self._client.beta.messages.stream(
                **self._request(messages, system, max_tokens)
            ) as stream:
                for text in stream.text_stream:
                    sink.text_parts.append(text)
                    yield text
                final = stream.get_final_message()
        except anthropic.APIError as exc:
            log.warning("llm_stream_failed", provider="anthropic", error=str(exc))
            raise LLMError(f"Anthropic stream failed: {exc}") from exc

        if final.stop_reason == "refusal":
            raise LLMError("Anthropic declined the request")
        sink.model = final.model
        sink.tokens_in = final.usage.input_tokens
        sink.tokens_out = final.usage.output_tokens
