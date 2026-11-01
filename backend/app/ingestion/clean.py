"""Text cleaning: repeated headers and footers, hyphenation, unicode and whitespace."""

import re
import unicodedata
from collections import Counter

from app.ingestion.types import Block

# NFKC would also fix ligatures, but it rewrites x² as x2 and so corrupts formulas.
_CHARACTER_MAP = str.maketrans(
    {
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "ﬅ": "st",
        "ﬆ": "st",
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        " ": " ",
        "­": "",
        "​": "",
        "‌": "",
        "‍": "",
        "﻿": "",
    }
)
_PAGE_NUMBER_LINE = re.compile(
    r"^[\W\d]*$"  # 12, - 12 -, 3/9
    r"|^page\s+\d+"  # Page 3, Page 3 of 9
    r"|\bpage\s+\d+\s*(of|/)\s*\d+\W*$"  # Syllabus page 3 of 9
    r"|[|•·:–—-]\s*page\s+\d+\W*$",  # CSE Department | Page 3
    re.IGNORECASE,
)
_LIST_ITEM = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[a-z][.)])\s")
_MIN_PAGES_FOR_REPEAT_CHECK = 3


def _line_key(line: str) -> str:
    """Return a comparison key for a line.

    Digits are ignored only on page-number lines, so "Page 3 of 9" matches "Page 4 of 9".
    "Unit 1" and "Unit 2" stay distinct, because they are real headings.
    """
    key = re.sub(r"\s+", " ", line).strip().lower()
    return re.sub(r"\d+", "#", key) if _PAGE_NUMBER_LINE.search(key) else key


def remove_repeated_lines(blocks: list[Block], threshold: float = 0.6) -> list[Block]:
    """Drop lines that appear on more than `threshold` of pages. Tables are left alone."""
    pages = {block.page for block in blocks if block.page is not None}
    if len(pages) < _MIN_PAGES_FOR_REPEAT_CHECK:
        return blocks

    pages_with_line: Counter[str] = Counter()
    seen: set[tuple[int, str]] = set()
    for block in blocks:
        if block.is_table or block.page is None:
            continue
        for line in block.text.splitlines():
            key = _line_key(line)
            if key and (block.page, key) not in seen:
                seen.add((block.page, key))
                pages_with_line[key] += 1

    limit = threshold * len(pages)
    repeated = {key for key, count in pages_with_line.items() if count > limit}
    if not repeated:
        return blocks

    kept = []
    for block in blocks:
        if block.is_table:
            kept.append(block)
            continue
        lines = [line for line in block.text.splitlines() if _line_key(line) not in repeated]
        if lines:
            kept.append(Block("\n".join(lines), block.page, block.heading_level, block.is_table))
    return kept


def normalize_unicode(text: str) -> str:
    """Compose characters and replace ligatures, smart punctuation and invisible characters."""
    return unicodedata.normalize("NFC", text).translate(_CHARACTER_MAP)


def fix_hyphenation(text: str) -> str:
    """Rejoin words split across lines, as in "regula-\\ntion".

    Only lowercase continuations are joined, so "Covid-\\n19" keeps its hyphen.
    """
    return re.sub(r"(?<=[a-z])-[ \t]*\n[ \t]*(?=[a-z])", "", text)


def unwrap_lines(text: str) -> str:
    """Join lines wrapped by the page layout, keeping paragraph breaks and list items."""
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        merged: list[str] = []
        for line in paragraph.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if merged and not _LIST_ITEM.match(line):
                merged[-1] = f"{merged[-1]} {stripped}"
            else:
                merged.append(stripped)
        if merged:
            paragraphs.append("\n".join(merged))
    return "\n\n".join(paragraphs)


def clean_text(text: str) -> str:
    """Apply all text-level cleaning."""
    text = fix_hyphenation(normalize_unicode(text))
    text = re.sub(r"[ \t]+", " ", text)
    return unwrap_lines(text).strip()


def clean_blocks(blocks: list[Block], threshold: float = 0.6) -> list[Block]:
    """Clean a document's blocks. Tables keep their layout and only get unicode fixes."""
    cleaned = []
    for block in remove_repeated_lines(blocks, threshold):
        text = normalize_unicode(block.text) if block.is_table else clean_text(block.text)
        if text:
            cleaned.append(Block(text, block.page, block.heading_level, block.is_table))
    return cleaned
