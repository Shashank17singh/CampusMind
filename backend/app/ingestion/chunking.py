"""Recursive, heading-aware chunking.

Sections are never merged across a heading, tables always get their own chunk, and long
paragraphs are split at the largest boundary that fits: paragraph, line, sentence, word.
"""

import re

from app.ingestion.tokens import count_tokens, decode, encode
from app.ingestion.types import Block, Chunk
from app.logging import get_logger

log = get_logger(__name__)

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_SEPARATORS: tuple[str | re.Pattern[str], ...] = ("\n\n", "\n", _SENTENCE_BREAK, " ")
_PARAGRAPH_JOIN = "\n\n"


def _split_on(text: str, separator: str | re.Pattern[str]) -> list[str]:
    """Split text and drop empty pieces."""
    pieces = separator.split(text) if isinstance(separator, re.Pattern) else text.split(separator)
    return [piece.strip() for piece in pieces if piece.strip()]


def split_text(text: str, limit: int, separators: int = 0) -> list[str]:
    """Split `text` into pieces of at most `limit` tokens, preferring natural boundaries."""
    if count_tokens(text) <= limit:
        return [text]
    if separators >= len(_SEPARATORS):
        tokens = encode(text)
        return [decode(tokens[start : start + limit]) for start in range(0, len(tokens), limit)]

    pieces = _split_on(text, _SEPARATORS[separators])
    if len(pieces) <= 1:
        return split_text(text, limit, separators + 1)

    joiner = _SEPARATORS[separators]
    joiner = joiner if isinstance(joiner, str) else " "
    result: list[str] = []
    current = ""
    for piece in pieces:
        candidate = f"{current}{joiner}{piece}" if current else piece
        if count_tokens(candidate) <= limit:
            current = candidate
            continue
        if current:
            result.append(current)
        if count_tokens(piece) <= limit:
            current = piece
        else:
            result.extend(split_text(piece, limit, separators + 1))
            current = ""
    if current:
        result.append(current)
    return result


def overlap_tail(text: str, overlap: int) -> str:
    """Return up to `overlap` tokens from the end of `text`, starting at a sentence if possible."""
    if overlap <= 0:
        return ""
    tail = ""
    for sentence in reversed(_split_on(text, _SENTENCE_BREAK)):
        candidate = f"{sentence} {tail}".strip()
        if count_tokens(candidate) > overlap:
            break
        tail = candidate
    if tail:
        return tail
    fragment = decode(encode(text)[-overlap:])
    _, _, after_first_space = fragment.partition(" ")
    return after_first_space or fragment


class _ChunkBuilder:
    """Accumulates pieces into chunks and seeds each new chunk with overlap."""

    def __init__(self, size: int, overlap: int) -> None:
        self.size = size
        self.overlap = overlap
        self.chunks: list[Chunk] = []
        self._parts: list[str] = []
        self._seed = ""
        self._page: int | None = None
        self._heading: str | None = None

    def add(self, piece: str, page: int | None, heading: str | None) -> None:
        """Add a piece, which must already fit within size minus overlap."""
        if self._parts and count_tokens(_PARAGRAPH_JOIN.join([*self._parts, piece])) > self.size:
            self.flush(keep_overlap=True)
        if not self._parts:
            self._page = page
            self._heading = heading
            if self._seed and count_tokens(f"{self._seed}{_PARAGRAPH_JOIN}{piece}") <= self.size:
                self._parts.append(self._seed)
            self._seed = ""
        self._parts.append(piece)

    def flush(self, keep_overlap: bool = False) -> None:
        """Emit the current chunk. With `keep_overlap`, its tail seeds the next chunk."""
        if not self._parts:
            return
        text = _PARAGRAPH_JOIN.join(self._parts)
        self._emit(text, self._page, self._heading, is_table=False)
        self._seed = overlap_tail(text, self.overlap) if keep_overlap else ""
        self._parts = []

    def add_table(self, text: str, page: int | None, heading: str | None) -> None:
        """Emit a table as its own chunk, even if it is over the size limit."""
        self.flush()
        tokens = count_tokens(text)
        if tokens > self.size:
            log.info("oversized_table_kept_whole", tokens=tokens, limit=self.size)
        self._emit(text, page, heading, is_table=True)

    def _emit(self, text: str, page: int | None, heading: str | None, *, is_table: bool) -> None:
        """Record a finished chunk."""
        self.chunks.append(
            Chunk(
                text=text,
                token_count=count_tokens(text),
                chunk_index=len(self.chunks),
                page_number=page,
                section_heading=heading,
                is_table=is_table,
            )
        )


def chunk_blocks(blocks: list[Block], size: int = 800, overlap: int = 120) -> list[Chunk]:
    """Turn cleaned blocks into chunks of at most `size` tokens. Tables may exceed it."""
    builder = _ChunkBuilder(size, overlap)
    stack: list[tuple[int, str]] = []
    heading: str | None = None

    for block in blocks:
        if block.heading_level is not None:
            builder.flush()
            while stack and stack[-1][0] >= block.heading_level:
                stack.pop()
            stack.append((block.heading_level, block.text.replace("\n", " ")))
            heading = " > ".join(text for _, text in stack)
        elif block.is_table:
            builder.add_table(block.text, block.page, heading)
        else:
            for piece in split_text(block.text, size - overlap):
                builder.add(piece, block.page, heading)

    builder.flush()
    return builder.chunks
