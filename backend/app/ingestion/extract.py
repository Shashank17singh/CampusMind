"""Text and table extraction for each supported file type."""

import io
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pdfplumber
import pymupdf
from docx import Document as DocxDocument
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph
from PIL import Image, UnidentifiedImageError
from pptx import Presentation
from pptx.enum.shapes import PP_PLACEHOLDER

from app.ingestion.errors import CorruptedFileError, ExtractionError, OcrUnavailableError
from app.ingestion.types import Block, ExtractedDocument, OcrEngine
from app.ingestion.validation import FileKind
from app.logging import get_logger

log = get_logger(__name__)

HEADING_PATTERN = re.compile(
    r"^(unit|chapter|module|section|part|annexure|appendix)\s+([0-9]+|[ivxlc]+)\b",
    re.IGNORECASE,
)
HEADING_SIZE_RATIO = 1.15
MAX_HEADING_CHARS = 150
MAX_BOLD_HEADING_CHARS = 80


def table_to_markdown(rows: list[list[Any]]) -> str:
    """Convert table rows to Markdown, treating the first row as the header."""
    cleaned = [
        [re.sub(r"\s+", " ", str(cell or "")).strip().replace("|", "\\|") for cell in row]
        for row in rows
    ]
    width = max(len(row) for row in cleaned)
    padded = [row + [""] * (width - len(row)) for row in cleaned]
    lines = ["| " + " | ".join(padded[0]) + " |", "|" + " --- |" * width]
    lines.extend("| " + " | ".join(row) + " |" for row in padded[1:])
    return "\n".join(lines)


def _paragraph_blocks(text: str, page: int | None) -> list[Block]:
    """Split plain text on blank lines, spotting headings by pattern or by shape."""
    blocks = []
    for paragraph in re.split(r"\n\s*\n", text):
        stripped = paragraph.strip()
        if not stripped:
            continue
        first_line = stripped.splitlines()[0]
        is_short = "\n" not in stripped and len(stripped) <= MAX_BOLD_HEADING_CHARS
        if HEADING_PATTERN.match(first_line) and is_short:
            blocks.append(Block(stripped, page, heading_level=2))
        elif is_short and stripped.isupper() and len(stripped) > 3:
            blocks.append(Block(stripped, page, heading_level=1))
        else:
            blocks.append(Block(stripped, page))
    return blocks


def _pdf_tables(path: Path) -> dict[int, list[tuple[tuple[float, ...], str]]]:
    """Return each page's tables as (bounding box, Markdown), keyed by page number."""
    tables: dict[int, list[tuple[tuple[float, ...], str]]] = {}
    with pdfplumber.open(path) as pdf:
        for number, page in enumerate(pdf.pages, start=1):
            for table in page.find_tables():
                rows = table.extract()
                if len(rows) >= 2 and len(rows[0]) >= 2:
                    tables.setdefault(number, []).append(
                        (tuple(table.bbox), table_to_markdown(rows))
                    )
    return tables


def _inside(block_box: tuple[float, ...], table_box: tuple[float, ...]) -> bool:
    """Return True if a text block's centre falls within a table's box."""
    x = (block_box[0] + block_box[2]) / 2
    y = (block_box[1] + block_box[3]) / 2
    return table_box[0] <= x <= table_box[2] and table_box[1] <= y <= table_box[3]


def _body_font_size(pages: list[dict[str, Any]]) -> float:
    """Return the font size carrying the most characters across the document."""
    sizes: Counter[float] = Counter()
    for page in pages:
        for block in page["blocks"]:
            for line in block.get("lines", []):
                for span in line["spans"]:
                    sizes[round(span["size"], 1)] += len(span["text"].strip())
    return sizes.most_common(1)[0][0] if sizes else 0.0


def _heading_level(
    block: dict[str, Any], text: str, body_size: float, levels: dict[float, int]
) -> int | None:
    """Classify a PDF text block as a heading from its size, weight and wording."""
    lines = block.get("lines", [])
    if len(lines) > 2 or not text or len(text) > MAX_HEADING_CHARS:
        return None
    spans = [span for line in lines for span in line["spans"] if span["text"].strip()]
    if not spans:
        return None
    size = round(max(span["size"] for span in spans), 1)
    if size >= body_size * HEADING_SIZE_RATIO:
        return levels.get(size, 3)
    is_bold = all(span["flags"] & 16 for span in spans)
    if is_bold and len(text) <= MAX_BOLD_HEADING_CHARS and not text.endswith("."):
        return 3
    return 2 if HEADING_PATTERN.match(text) else None


def _block_text(block: dict[str, Any]) -> str:
    """Join a PDF text block's spans into lines."""
    lines = ("".join(span["text"] for span in line["spans"]) for line in block.get("lines", []))
    return "\n".join(line for line in lines if line.strip()).strip()


def _ocr_pdf_page(page: Any, number: int, ocr: OcrEngine, dpi: int) -> list[Block]:
    """Render a PDF page to an image and OCR it."""
    pixmap = page.get_pixmap(dpi=dpi)
    image = Image.open(io.BytesIO(pixmap.tobytes("png")))
    return _paragraph_blocks(ocr.image_to_text(image), number)


def extract_pdf(path: Path, ocr: OcrEngine, ocr_min_chars: int, dpi: int) -> ExtractedDocument:
    """Extract a PDF, falling back to OCR on pages with almost no text."""
    try:
        document = pymupdf.open(path)
    except Exception as exc:  # PyMuPDF raises several unrelated types on bad input.
        raise CorruptedFileError("This PDF is corrupted and cannot be read.") from exc

    with document:
        if document.needs_pass:
            raise CorruptedFileError("This PDF is password protected. Remove the password first.")
        if document.page_count == 0:
            raise CorruptedFileError("This PDF has no pages.")

        try:
            tables = _pdf_tables(path)
            pages = list(document.pages())
            page_dicts = [page.get_text("dict", sort=True) for page in pages]
        except Exception as exc:
            raise CorruptedFileError("This PDF is corrupted and cannot be read.") from exc

        body_size = _body_font_size(page_dicts)
        heading_sizes = sorted(
            {
                round(span["size"], 1)
                for page in page_dicts
                for block in page["blocks"]
                for line in block.get("lines", [])
                for span in line["spans"]
                if span["size"] >= body_size * HEADING_SIZE_RATIO and span["text"].strip()
            },
            reverse=True,
        )
        levels = {size: min(index + 1, 3) for index, size in enumerate(heading_sizes)}

        blocks: list[Block] = []
        warnings: list[str] = []
        for number, (page, page_dict) in enumerate(zip(pages, page_dicts, strict=True), start=1):
            if len(page.get_text().strip()) < ocr_min_chars:
                try:
                    blocks.extend(_ocr_pdf_page(page, number, ocr, dpi))
                except OcrUnavailableError:
                    warnings.append(f"Page {number} needs OCR, which is unavailable.")
                continue

            page_tables = tables.get(number, [])
            positioned: list[tuple[float, Block]] = [
                (box[1], Block(markdown, number, is_table=True)) for box, markdown in page_tables
            ]
            for block in page_dict["blocks"]:
                if block.get("type") != 0:
                    continue
                box = tuple(block["bbox"])
                if any(_inside(box, table_box) for table_box, _ in page_tables):
                    continue
                text = _block_text(block)
                if text:
                    level = _heading_level(block, text, body_size, levels)
                    positioned.append((box[1], Block(text, number, heading_level=level)))
            positioned.sort(key=lambda item: item[0])
            blocks.extend(block for _, block in positioned)

        return ExtractedDocument(blocks, document.page_count, warnings)


def extract_docx(path: Path) -> ExtractedDocument:
    """Extract a Word file in body order, keeping tables in place."""
    try:
        document = DocxDocument(str(path))
    except Exception as exc:  # python-docx surfaces KeyError, ValueError and lxml errors.
        raise CorruptedFileError("This DOCX file is corrupted and cannot be read.") from exc

    blocks: list[Block] = []
    for item in document.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            style = item.style.name if item.style is not None and item.style.name else ""
            level: int | None = None
            if style == "Title":
                level = 1
            elif style.startswith("Heading"):
                digits = style.removeprefix("Heading").strip()
                level = int(digits) if digits.isdigit() else 2
            elif HEADING_PATTERN.match(text) and len(text) <= MAX_BOLD_HEADING_CHARS:
                level = 2
            blocks.append(Block(text, None, heading_level=level))
        elif isinstance(item, DocxTable):
            rows = [[cell.text for cell in row.cells] for row in item.rows]
            if rows:
                blocks.append(Block(table_to_markdown(rows), None, is_table=True))
    return ExtractedDocument(blocks, page_count=1)


def extract_pptx(path: Path) -> ExtractedDocument:
    """Extract slides, using each slide's title as a heading."""
    try:
        presentation = Presentation(str(path))
    except Exception as exc:  # python-pptx surfaces KeyError, ValueError and lxml errors.
        raise CorruptedFileError("This PPTX file is corrupted and cannot be read.") from exc

    blocks: list[Block] = []
    slides = list(presentation.slides)
    for number, slide in enumerate(slides, start=1):
        for shape in slide.shapes:
            if shape.has_table:
                rows = [[cell.text for cell in row.cells] for row in shape.table.rows]
                blocks.append(Block(table_to_markdown(rows), number, is_table=True))
            elif shape.has_text_frame and shape.text_frame.text.strip():
                is_title = shape.is_placeholder and shape.placeholder_format.type in (
                    PP_PLACEHOLDER.TITLE,
                    PP_PLACEHOLDER.CENTER_TITLE,
                )
                blocks.append(
                    Block(
                        shape.text_frame.text.strip(), number, heading_level=1 if is_title else None
                    )
                )
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame is not None:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                blocks.append(Block(notes, number))
    return ExtractedDocument(blocks, page_count=len(slides))


def extract_txt(path: Path) -> ExtractedDocument:
    """Extract a text file, trying UTF-8 before falling back to Windows-1252."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return ExtractedDocument(_paragraph_blocks(text, None), page_count=1)


def extract_image(path: Path, ocr: OcrEngine) -> ExtractedDocument:
    """OCR a photographed or scanned notice."""
    try:
        with Image.open(path) as image:
            image.load()
            text = ocr.image_to_text(image)
    except (UnidentifiedImageError, OSError) as exc:
        raise CorruptedFileError("This image is corrupted and cannot be read.") from exc
    return ExtractedDocument(_paragraph_blocks(text, 1), page_count=1)


def extract(
    path: Path, kind: FileKind, ocr: OcrEngine, ocr_min_chars: int = 50, dpi: int = 300
) -> ExtractedDocument:
    """Extract a file of any supported kind. Raises if no text at all was found."""
    if kind is FileKind.PDF:
        result = extract_pdf(path, ocr, ocr_min_chars, dpi)
    elif kind is FileKind.DOCX:
        result = extract_docx(path)
    elif kind is FileKind.PPTX:
        result = extract_pptx(path)
    elif kind is FileKind.TXT:
        result = extract_txt(path)
    else:
        result = extract_image(path, ocr)

    if not result.blocks:
        detail = " ".join(result.warnings) or "The file appears to be empty."
        raise ExtractionError(f"No text could be extracted. {detail}")
    if result.warnings:
        log.warning("extraction_warnings", path=str(path), warnings=result.warnings)
    return result
