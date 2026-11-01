"""Token counting shared by the chunker and the RAG context limit."""

from functools import lru_cache

import tiktoken


@lru_cache
def _encoding() -> tiktoken.Encoding:
    """Load the tokenizer once."""
    return tiktoken.get_encoding("cl100k_base")


def encode(text: str) -> list[int]:
    """Tokenize text. Text that looks like a special token is treated as plain text."""
    return _encoding().encode(text, disallowed_special=())


def decode(tokens: list[int]) -> str:
    """Turn tokens back into text."""
    return _encoding().decode(tokens)


def count_tokens(text: str) -> int:
    """Return the number of tokens in `text`."""
    return len(encode(text))
