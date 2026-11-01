"""The provider-neutral LLM interface used by every agent."""

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from pydantic import BaseModel, ValidationError

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMError(Exception):
    """The LLM call failed or returned something unusable."""


@dataclass(frozen=True)
class ChatMessage:
    """One message in a conversation."""

    role: Literal["user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    """A completed LLM call and its token usage."""

    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class StreamResult:
    """Filled in by `stream` once the final token has been yielded."""

    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    text_parts: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        """The full text streamed so far."""
        return "".join(self.text_parts)


def extract_json(text: str) -> Any:
    """Parse the first JSON value in `text`, tolerating code fences and surrounding prose."""
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(stripped):
        if char in "{[":
            try:
                value, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            return value
    raise LLMError("LLM reply contained no JSON")


class LLMClient(ABC):
    """Text completion, streaming and schema-validated JSON, independent of provider."""

    @abstractmethod
    def complete(
        self,
        messages: list[ChatMessage],
        *,
        system: str = "",
        max_tokens: int = 1024,
        json_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Return a full reply. With `json_schema`, the reply should be JSON matching it."""

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        *,
        system: str = "",
        max_tokens: int = 2048,
        result: StreamResult | None = None,
    ) -> Iterator[str]:
        """Yield text fragments. Usage is written to `result` when the stream ends."""

    def complete_json(
        self,
        messages: list[ChatMessage],
        schema: type[SchemaT],
        *,
        system: str = "",
        max_tokens: int = 1024,
    ) -> tuple[SchemaT, LLMResponse]:
        """Return a reply parsed into `schema`, or raise LLMError if it does not validate."""
        response = self.complete(
            messages,
            system=system,
            max_tokens=max_tokens,
            json_schema=schema.model_json_schema(),
        )
        try:
            return schema.model_validate(extract_json(response.text)), response
        except ValidationError as exc:
            raise LLMError(f"LLM reply failed {schema.__name__} validation: {exc}") from exc
