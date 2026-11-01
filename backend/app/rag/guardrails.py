"""Guardrails: prompt injection in documents, personal data in answers, and context size."""

import re
from collections.abc import Iterable, Iterator

from app.ingestion.tokens import count_tokens
from app.logging import get_logger
from app.vectorstore import ScoredChunk

log = get_logger(__name__)

INJECTION_REMOVED = "[text removed: it looked like instructions to the assistant]"
PHONE_REMOVED = "[phone number removed]"
EMAIL_REMOVED = "[email removed]"

_INJECTION = re.compile(
    r"(?:ignore|disregard|forget|override)\b[^.\n]{0,40}"
    r"\b(?:instructions?|prompts?|rules?|context|above)\b"
    r"|\byou\s+are\s+now\b"
    r"|\bact\s+as\s+(?:an?\s+)?(?:unrestricted|different|new)\b"
    r"|\b(?:reveal|show|print|repeat)\b[^.\n]{0,30}\b(?:system\s+prompt|instructions)\b"
    r"|\bnew\s+instructions\s*:"
    r"|\bsystem\s*:\s",
    re.IGNORECASE,
)
_PHONE = re.compile(
    r"(?<![\w.])(?:\+?91[\s-]?)?(?:[6-9]\d{4}[\s-]?\d{5}|0\d{2,4}[\s-]\d{6,8})(?!\d)"
)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")

_SOURCE_TAG = re.compile(r"</?source\b[^>]*>\s*", re.IGNORECASE)

# Longer than any phone number, email or source tag worth catching, so one split across
# stream fragments is always still in the buffer when its last character arrives.
_HOLD_BACK_CHARS = 200


def neutralize_injection(text: str) -> str:
    """Replace sentences that try to give the assistant orders.

    This is one layer of two. The system prompt also tells the model that sources are
    data, because no pattern list can catch every way of phrasing an attack.
    """
    cleaned, removed = _INJECTION.subn(INJECTION_REMOVED, text)
    if removed:
        log.warning("prompt_injection_neutralized", occurrences=removed)
    return cleaned.replace("</source>", "")


def _digits(text: str) -> str:
    """Return only the last ten digits, so +91 98765 43210 matches 9876543210."""
    return re.sub(r"\D", "", text)[-10:]


def allowed_contacts(chunks: Iterable[ScoredChunk]) -> set[str]:
    """Collect the numbers and emails printed in official contact lists among the sources."""
    allowed: set[str] = set()
    for chunk in chunks:
        if chunk.doc_type == "contact_list":
            allowed.update(_digits(match) for match in _PHONE.findall(chunk.text))
            allowed.update(match.lower() for match in _EMAIL.findall(chunk.text))
    return allowed


def redact(text: str, allowed: set[str]) -> str:
    """Remove phone numbers and emails, except those from an official contact list.

    Prompt markup goes too, because small models sometimes echo their sources.
    """

    def replace_phone(match: re.Match[str]) -> str:
        return match.group() if _digits(match.group()) in allowed else PHONE_REMOVED

    def replace_email(match: re.Match[str]) -> str:
        return match.group() if match.group().lower() in allowed else EMAIL_REMOVED

    return _EMAIL.sub(replace_email, _PHONE.sub(replace_phone, _SOURCE_TAG.sub("", text)))


def redact_stream(fragments: Iterable[str], allowed: set[str]) -> Iterator[str]:
    """Redact a token stream.

    A number can arrive across several fragments, so the end of the text is held back
    until enough has followed it to be sure it is not the start of a number.
    """
    buffer = ""
    for fragment in fragments:
        buffer = redact(buffer + fragment, allowed)
        if len(buffer) > _HOLD_BACK_CHARS:
            cut = len(buffer) - _HOLD_BACK_CHARS
            yield buffer[:cut]
            buffer = buffer[cut:]
    if buffer:
        yield redact(buffer, allowed)


def limit_context(chunks: list[ScoredChunk], max_tokens: int) -> list[ScoredChunk]:
    """Keep the best-ranked chunks that fit in the budget, skipping any that alone cannot."""
    kept = []
    used = 0
    for chunk in chunks:
        tokens = count_tokens(chunk.text)
        if used + tokens <= max_tokens:
            kept.append(chunk)
            used += tokens
    return kept
