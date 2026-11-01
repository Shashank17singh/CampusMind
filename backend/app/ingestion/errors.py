"""Ingestion failures. Each message is safe to show to the uploader."""


class IngestionError(Exception):
    """Base class. Stops the pipeline and marks the document as failed."""

    code = "ingestion_failed"
    status_code = 422


class UnsupportedFileError(IngestionError):
    """The file type is not accepted."""

    code = "unsupported_file_type"
    status_code = 415


class FileTooLargeError(IngestionError):
    """The file is over the size limit."""

    code = "file_too_large"
    status_code = 413


class CorruptedFileError(IngestionError):
    """The file could not be parsed as its claimed type."""

    code = "corrupted_file"
    status_code = 422


class ExtractionError(IngestionError):
    """The file parsed but no usable text came out."""

    code = "extraction_failed"
    status_code = 422


class OcrUnavailableError(IngestionError):
    """OCR is needed but the Tesseract binary is not installed."""

    code = "ocr_unavailable"
    status_code = 503
