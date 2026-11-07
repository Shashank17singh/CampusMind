"""LLM clients, selected by the LLM_PROVIDER environment variable."""

from functools import lru_cache

from app.config import get_settings
from app.llm.base import ChatMessage, LLMClient, LLMError, LLMResponse, StreamResult

__all__ = ["ChatMessage", "LLMClient", "LLMError", "LLMResponse", "StreamResult", "get_llm"]


@lru_cache
def get_llm() -> LLMClient:
    """Return the configured client, importing only the provider that is in use."""
    settings = get_settings()
    if settings.llm_provider == "anthropic":
        from app.llm.anthropic_client import AnthropicClient

        return AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.resolved_llm_model,
            timeout=settings.llm_timeout_seconds,
        )
    if settings.llm_provider == "groq":
        from app.llm.groq_client import GroqClient

        return GroqClient(
            api_key=settings.groq_api_key,
            model=settings.resolved_llm_model,
            timeout=settings.llm_timeout_seconds,
        )

    from app.llm.ollama_client import OllamaClient

    return OllamaClient(
        base_url=settings.ollama_base_url,
        model=settings.resolved_llm_model,
        timeout=settings.llm_timeout_seconds,
    )
