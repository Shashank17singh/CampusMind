"""Typed tools. Arguments are validated with Pydantic before a tool runs.

LLMs produce tool arguments, so they are untrusted input and get the same treatment as a
request body. A malformed call never reaches the function behind the tool.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ValidationError

from app.logging import get_logger

log = get_logger(__name__)

ArgsT = TypeVar("ArgsT", bound=BaseModel)


class ToolError(Exception):
    """A tool was unknown, or was called with invalid arguments."""


@dataclass(frozen=True)
class Tool(Generic[ArgsT]):
    """A function an agent can call, and the model describing its arguments."""

    name: str
    description: str
    args: type[ArgsT]
    function: Callable[[ArgsT], Any]

    def json_schema(self) -> dict[str, Any]:
        """Describe the tool for an LLM prompt."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args.model_json_schema(),
        }


class ToolRegistry:
    """The tools one agent exposes."""

    def __init__(self, tools: list[Tool[Any]]) -> None:
        """Index tools by name."""
        self._tools = {tool.name: tool for tool in tools}

    def call(self, name: str, raw_args: dict[str, Any]) -> Any:
        """Validate the arguments, then run the tool."""
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"Unknown tool: {name}")
        try:
            args = tool.args.model_validate(raw_args)
        except ValidationError as exc:
            log.warning("tool_arguments_rejected", tool=name, errors=exc.error_count())
            raise ToolError(f"Invalid arguments for {name}: {exc}") from exc
        return tool.function(args)

    def schemas(self) -> list[dict[str, Any]]:
        """Describe every tool."""
        return [tool.json_schema() for tool in self._tools.values()]
