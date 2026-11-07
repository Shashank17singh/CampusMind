"""Tests for the LLM clients, using stubbed transports."""

import json
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from app.config import Settings
from app.llm import ChatMessage, LLMError, StreamResult
from app.llm.anthropic_client import AnthropicClient, _strict_schema
from app.llm.base import extract_json
from app.llm.groq_client import GroqClient
from app.llm.ollama_client import OllamaClient

MESSAGES = [ChatMessage("user", "hello")]


class Answer(BaseModel):
    value: int


class TestExtractJson:
    @pytest.mark.parametrize(
        "text",
        [
            '{"value": 1}',
            '```json\n{"value": 1}\n```',
            'Sure! Here it is: {"value": 1} Hope that helps.',
        ],
    )
    def test_finds_json_in_common_reply_shapes(self, text: str) -> None:
        assert extract_json(text) == {"value": 1}

    def test_raises_when_there_is_no_json(self) -> None:
        with pytest.raises(LLMError):
            extract_json("no structured data here")


def ollama(handler: Any) -> OllamaClient:
    """Build a client whose HTTP calls go to `handler`."""
    return OllamaClient("http://ollama", "llama", transport=httpx.MockTransport(handler))


class TestOllama:
    def test_complete_sends_the_system_prompt_and_reads_usage(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            reply = {
                "model": "llama",
                "message": {"content": "hi"},
                "prompt_eval_count": 7,
                "eval_count": 3,
            }
            return httpx.Response(200, json=reply)

        response = ollama(handler).complete(MESSAGES, system="Be brief", max_tokens=50)
        assert seen["messages"][0] == {"role": "system", "content": "Be brief"}
        assert seen["options"]["num_predict"] == 50
        assert (response.text, response.tokens_in, response.tokens_out) == ("hi", 7, 3)

    def test_json_schema_goes_in_the_format_field(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json={"message": {"content": '{"value": 5}'}})

        parsed, _ = ollama(handler).complete_json(MESSAGES, Answer)
        assert parsed.value == 5
        assert seen["format"]["properties"]["value"]["type"] == "integer"

    def test_schema_mismatch_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"message": {"content": '{"value": "many"}'}})

        with pytest.raises(LLMError, match="validation"):
            ollama(handler).complete_json(MESSAGES, Answer)

    def test_server_error_raises_llm_error(self) -> None:
        with pytest.raises(LLMError):
            ollama(lambda _: httpx.Response(500)).complete(MESSAGES)

    def test_connection_failure_raises_llm_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        with pytest.raises(LLMError):
            ollama(handler).complete(MESSAGES)

    def test_stream_yields_fragments_and_records_usage(self) -> None:
        lines = [
            {"message": {"content": "Hel"}, "done": False},
            {"message": {"content": "lo"}, "done": False},
            {"message": {"content": ""}, "done": True, "model": "llama", "eval_count": 2},
        ]
        body = "\n".join(json.dumps(line) for line in lines)
        result = StreamResult()
        client = ollama(lambda _: httpx.Response(200, text=body))
        assert list(client.stream(MESSAGES, result=result)) == ["Hel", "lo"]
        assert (result.text, result.tokens_out) == ("Hello", 2)


def anthropic_reply(text: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    """Build an object shaped like an SDK Message."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        model="claude-opus-5-5",
        stop_reason=stop_reason,
        usage=SimpleNamespace(input_tokens=11, output_tokens=4),
    )


class FakeStream:
    """Stands in for the SDK's stream context manager."""

    def __init__(self, reply: SimpleNamespace, fragments: list[str]) -> None:
        self.text_stream = iter(fragments)
        self._reply = reply

    def __enter__(self) -> "FakeStream":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def get_final_message(self) -> SimpleNamespace:
        return self._reply


def claude(messages: Any, model: str = "claude-opus-5-5") -> AnthropicClient:
    """Build a client around a stub of the SDK's `beta.messages` resource."""
    stub = SimpleNamespace(beta=SimpleNamespace(messages=messages))
    return AnthropicClient("key", model, client=stub)  # type: ignore[arg-type]


class TestAnthropic:
    def test_complete_skips_non_text_blocks_and_reads_usage(self) -> None:
        calls: list[dict[str, Any]] = []

        def create(**request: Any) -> SimpleNamespace:
            calls.append(request)
            return anthropic_reply("hello")

        response = claude(SimpleNamespace(create=create)).complete(MESSAGES, system="Be brief")
        assert (response.text, response.tokens_in, response.tokens_out) == ("hello", 11, 4)
        assert calls[0]["system"] == "Be brief"
        assert "temperature" not in calls[0]

    def test_refusal_fallback_is_only_sent_to_models_that_accept_it(self) -> None:
        calls: list[dict[str, Any]] = []

        def create(**request: Any) -> SimpleNamespace:
            calls.append(request)
            return anthropic_reply("hi")

        claude(SimpleNamespace(create=create)).complete(MESSAGES)
        claude(SimpleNamespace(create=create), "claude-haiku-5-5").complete(MESSAGES)
        assert calls[0]["fallbacks"] == "default"
        assert calls[0]["betas"] == ["server-side-fallback-2026-07-01"]
        assert "fallbacks" not in calls[1]
        assert "betas" not in calls[1]

    def test_json_schema_becomes_a_strict_output_format(self) -> None:
        calls: list[dict[str, Any]] = []

        def create(**request: Any) -> SimpleNamespace:
            calls.append(request)
            return anthropic_reply('{"value": 3}')

        parsed, _ = claude(SimpleNamespace(create=create)).complete_json(MESSAGES, Answer)
        assert parsed.value == 3
        schema = calls[0]["output_config"]["format"]["schema"]
        assert schema["additionalProperties"] is False
        assert schema["required"] == ["value"]

    def test_refusal_raises(self) -> None:
        create = lambda **_: anthropic_reply("", stop_reason="refusal")  # noqa: E731
        with pytest.raises(LLMError, match="declined"):
            claude(SimpleNamespace(create=create)).complete(MESSAGES)

    def test_api_errors_become_llm_errors(self) -> None:
        def create(**_: Any) -> None:
            raise anthropic.APIConnectionError(request=None)  # type: ignore[arg-type]

        with pytest.raises(LLMError):
            claude(SimpleNamespace(create=create)).complete(MESSAGES)

    def test_stream_yields_text_and_records_usage(self) -> None:
        stream = lambda **_: FakeStream(anthropic_reply("Hello"), ["Hel", "lo"])  # noqa: E731
        result = StreamResult()
        fragments = list(claude(SimpleNamespace(stream=stream)).stream(MESSAGES, result=result))
        assert fragments == ["Hel", "lo"]
        assert (result.text, result.tokens_in, result.model) == ("Hello", 11, "claude-opus-5-5")

    def test_stream_refusal_raises(self) -> None:
        stream = lambda **_: FakeStream(anthropic_reply("", "refusal"), [])  # noqa: E731
        with pytest.raises(LLMError, match="declined"):
            list(claude(SimpleNamespace(stream=stream)).stream(MESSAGES))


def test_strict_schema_handles_nested_objects() -> None:
    class Inner(BaseModel):
        name: str

    class Outer(BaseModel):
        inner: Inner
        items: list[Inner]

    schema = _strict_schema(Outer.model_json_schema())
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Inner"]["additionalProperties"] is False
    assert "title" not in schema


def groq(handler: Any, model: str = "openai/gpt-oss-120b", retries: int = 2) -> GroqClient:
    """Build a Groq client whose HTTP calls go to `handler`, without real waiting."""
    return GroqClient(
        "test-key",
        model,
        transport=httpx.MockTransport(handler),
        max_retries=retries,
        sleep=lambda _: None,
    )


def groq_reply(content: str, prompt: int = 11, completion: int = 4) -> dict[str, Any]:
    return {
        "model": "openai/gpt-oss-120b",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion},
    }


class TestGroq:
    def test_complete_sends_key_system_prompt_and_reads_usage(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers["authorization"]
            seen["url"] = str(request.url)
            seen["body"] = json.loads(request.content)
            return httpx.Response(200, json=groq_reply("hi"))

        response = groq(handler).complete(MESSAGES, system="Be brief", max_tokens=50)
        body = seen["body"]
        assert seen["auth"] == "Bearer test-key"
        assert seen["url"].endswith("/openai/v1/chat/completions")
        assert body["messages"] == [
            {"role": "system", "content": "Be brief"},
            {"role": "user", "content": "hello"},
        ]
        assert body["max_completion_tokens"] == 50
        assert body["reasoning_effort"] == "low"
        assert body["include_reasoning"] is False
        assert (response.text, response.tokens_in, response.tokens_out) == ("hi", 11, 4)

    def test_reasoning_options_are_only_sent_to_gpt_oss(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=groq_reply("hi"))

        groq(handler, model="llama-3.3-70b-versatile").complete(MESSAGES)
        assert "reasoning_effort" not in seen
        assert "include_reasoning" not in seen

    def test_json_mode_puts_the_schema_in_the_prompt(self) -> None:
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, json=groq_reply('{"value": 5}'))

        parsed, _ = groq(handler).complete_json(MESSAGES, Answer, system="Count.")
        assert parsed.value == 5
        assert seen["response_format"] == {"type": "json_object"}
        system = seen["messages"][0]["content"]
        assert system.startswith("Count.")
        assert '"value"' in system and "JSON" in system

    def test_rate_limit_is_retried_after_the_given_wait(self) -> None:
        waits: list[float] = []
        calls = iter(
            [
                httpx.Response(429, headers={"retry-after": "3"}),
                httpx.Response(200, json=groq_reply("ok")),
            ]
        )
        client = GroqClient(
            "k",
            "openai/gpt-oss-120b",
            transport=httpx.MockTransport(lambda _: next(calls)),
            sleep=waits.append,
        )
        assert client.complete(MESSAGES).text == "ok"
        assert waits == [3.0]

    def test_rate_limit_gives_up_after_the_retries(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(429, headers={"retry-after": "1"})

        with pytest.raises(LLMError, match="429"):
            groq(handler, retries=2).complete(MESSAGES)
        assert len(attempts) == 3

    def test_client_errors_are_not_retried(self) -> None:
        attempts: list[int] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts.append(1)
            return httpx.Response(401, json={"error": {"message": "Invalid API Key"}})

        with pytest.raises(LLMError, match="Invalid API Key"):
            groq(handler).complete(MESSAGES)
        assert len(attempts) == 1

    def test_connection_failure_raises_llm_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        with pytest.raises(LLMError):
            groq(handler, retries=0).complete(MESSAGES)

    def test_stream_yields_fragments_and_records_usage(self) -> None:
        events = [
            {"model": "openai/gpt-oss-120b", "choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "x_groq": {"usage": {"prompt_tokens": 9, "completion_tokens": 2}},
            },
        ]
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"
        seen: dict[str, Any] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen.update(json.loads(request.content))
            return httpx.Response(200, text=body)

        result = StreamResult()
        assert list(groq(handler).stream(MESSAGES, result=result)) == ["Hel", "lo"]
        assert seen["stream"] is True
        assert (result.text, result.tokens_in, result.tokens_out) == ("Hello", 9, 2)
        assert result.model == "openai/gpt-oss-120b"

    def test_stream_failure_raises_llm_error(self) -> None:
        with pytest.raises(LLMError):
            list(groq(lambda _: httpx.Response(500), retries=0).stream(MESSAGES))


class TestGroqSettings:
    def test_groq_defaults_to_gpt_oss_120b(self) -> None:
        assert Settings(llm_provider="groq").resolved_llm_model == "openai/gpt-oss-120b"

    def test_production_needs_a_groq_key(self) -> None:
        secure = {"environment": "production", "secret_key": "s" * 40, "cookie_secure": True}
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            Settings(llm_provider="groq", **secure)
        Settings(llm_provider="groq", groq_api_key="gsk-test", **secure)
