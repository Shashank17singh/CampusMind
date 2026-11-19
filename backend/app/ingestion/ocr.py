"""Tesseract OCR."""

import statistics
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image

from app.ingestion.errors import OcrUnavailableError

ROW_TOLERANCE = 0.6
PARAGRAPH_GAP = 3.0
_DEFAULT_WIN_TESSERACT = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")


@dataclass(frozen=True)
class Word:
    """One word Tesseract found, with its box on the page."""

    text: str
    left: int
    top: int
    height: int

    @property
    def middle(self) -> float:
        """The word's vertical centre."""
        return self.top + self.height / 2


def join_rows(words: list[Word]) -> str:
    """Rebuild lines of text from word boxes, joining words that sit at the same height.

    Tesseract reads a scanned table column by column, which separates a date from the event
    beside it. Grouping words by their vertical centre puts each row back together. A gap of
    a few lines starts a new paragraph. The cost is that two columns of prose would be joined
    line by line, which scanned notices and tables rarely have.
    """
    if not words:
        return ""
    line_height = statistics.median(word.height for word in words)
    rows: list[list[Word]] = []
    centres: list[float] = []
    for word in sorted(words, key=lambda item: item.middle):
        if rows and abs(word.middle - centres[-1]) <= line_height * ROW_TOLERANCE:
            rows[-1].append(word)
            centres[-1] = statistics.fmean(item.middle for item in rows[-1])
        else:
            rows.append([word])
            centres.append(word.middle)

    lines: list[str] = []
    for index, row in enumerate(rows):
        if index and centres[index] - centres[index - 1] > line_height * PARAGRAPH_GAP:
            lines.append("")
        lines.append(" ".join(item.text for item in sorted(row, key=lambda item: item.left)))
    return "\n".join(lines)


class TesseractOcr:
    """OCR through the pytesseract wrapper. Needs the Tesseract binary on the machine."""

    def __init__(
        self,
        command: str = "",
        language: str = "hin+eng",
        tessdata_dir: str = "",
    ) -> None:
        """Set the binary path if given. Otherwise Tesseract is found through PATH or defaults."""
        if not command and _DEFAULT_WIN_TESSERACT.exists():
            command = str(_DEFAULT_WIN_TESSERACT)
        if command:
            pytesseract.pytesseract.tesseract_cmd = command
        self._language = language

        if not tessdata_dir:
            local_tessdata = Path(__file__).resolve().parents[2] / "data" / "tessdata"
            if local_tessdata.exists():
                tessdata_dir = str(local_tessdata)
        self._tessdata_dir = tessdata_dir
        if self._tessdata_dir:
            import os

            os.environ["TESSDATA_PREFIX"] = str(self._tessdata_dir)

    def image_to_text(self, image: Image.Image) -> str:
        """Return the text in `image`, with table rows kept on one line."""
        try:
            data = pytesseract.image_to_data(
                image,
                lang=self._language,
                output_type=pytesseract.Output.DICT,
            )
        except pytesseract.TesseractError:
            # Fallback to English if combined language packs are not both available
            try:
                data = pytesseract.image_to_data(
                    image,
                    lang="eng",
                    output_type=pytesseract.Output.DICT,
                )
            except Exception as exc:
                raise OcrUnavailableError("Tesseract OCR failed to process the image.") from exc
        except pytesseract.TesseractNotFoundError as exc:
            raise OcrUnavailableError(
                "This file needs OCR, but Tesseract is not installed on the server."
            ) from exc
        words = [
            Word(text.strip(), data["left"][i], data["top"][i], data["height"][i])
            for i, text in enumerate(data["text"])
            if text.strip() and float(data["conf"][i]) >= 0
        ]
        return join_rows(words)
