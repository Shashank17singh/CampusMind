"""Tesseract OCR."""

import pytesseract
from PIL import Image

from app.ingestion.errors import OcrUnavailableError


class TesseractOcr:
    """OCR through the pytesseract wrapper. Needs the Tesseract binary on the machine."""

    def __init__(self, command: str = "", language: str = "eng") -> None:
        """Set the binary path if given. Otherwise Tesseract is found through PATH."""
        if command:
            pytesseract.pytesseract.tesseract_cmd = command
        self._language = language

    def image_to_text(self, image: Image.Image) -> str:
        """Return the text in `image`."""
        try:
            return pytesseract.image_to_string(image, lang=self._language)
        except pytesseract.TesseractNotFoundError as exc:
            raise OcrUnavailableError(
                "This file needs OCR, but Tesseract is not installed on the server."
            ) from exc
