"""Upload validation. The extension must agree with the file's actual content."""

import io
import zipfile
from enum import StrEnum
from pathlib import PurePath

from app.ingestion.errors import CorruptedFileError, FileTooLargeError, UnsupportedFileError

MAX_ZIP_UNCOMPRESSED_BYTES = 250 * 1024 * 1024


class FileKind(StrEnum):
    """The accepted file families."""

    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    TXT = "txt"
    IMAGE = "image"


EXTENSIONS = {
    ".pdf": FileKind.PDF,
    ".docx": FileKind.DOCX,
    ".pptx": FileKind.PPTX,
    ".txt": FileKind.TXT,
    ".png": FileKind.IMAGE,
    ".jpg": FileKind.IMAGE,
    ".jpeg": FileKind.IMAGE,
}

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt": "text/plain",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

_ACCEPTED = "PDF, DOCX, PPTX, TXT, PNG or JPG"


def _check_zip_container(data: bytes, required_member: str, label: str) -> None:
    """Confirm an Office file is a sane zip holding its expected main part."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            names = set(archive.namelist())
            uncompressed = sum(info.file_size for info in archive.infolist())
    except zipfile.BadZipFile as exc:
        raise CorruptedFileError(f"This {label} file is corrupted and cannot be read.") from exc
    if required_member not in names:
        raise CorruptedFileError(f"This file is not a valid {label} document.")
    if uncompressed > MAX_ZIP_UNCOMPRESSED_BYTES:
        raise CorruptedFileError(f"This {label} file expands to an unsafe size.")


def _check_content(kind: FileKind, extension: str, data: bytes) -> None:
    """Compare the file's leading bytes with what its extension claims."""
    if kind is FileKind.PDF:
        if b"%PDF-" not in data[:1024]:
            raise CorruptedFileError("This file does not look like a valid PDF.")
    elif kind is FileKind.DOCX:
        _check_zip_container(data, "word/document.xml", "DOCX")
    elif kind is FileKind.PPTX:
        _check_zip_container(data, "ppt/presentation.xml", "PPTX")
    elif kind is FileKind.TXT:
        if b"\x00" in data[:8192]:
            raise UnsupportedFileError("This looks like a binary file, not plain text.")
    elif extension == ".png":
        if not data.startswith(b"\x89PNG\r\n\x1a\n"):
            raise CorruptedFileError("This file does not look like a valid PNG image.")
    elif not data.startswith(b"\xff\xd8\xff"):
        raise CorruptedFileError("This file does not look like a valid JPG image.")


def validate_upload(filename: str, data: bytes, max_bytes: int) -> tuple[FileKind, str]:
    """Return the file's kind and MIME type, or raise an IngestionError."""
    extension = PurePath(filename).suffix.lower()
    kind = EXTENSIONS.get(extension)
    if kind is None:
        raise UnsupportedFileError(f"Unsupported file type. Upload a {_ACCEPTED} file.")
    if not data:
        raise CorruptedFileError("The file is empty.")
    if len(data) > max_bytes:
        limit_mb = max_bytes // (1024 * 1024)
        raise FileTooLargeError(f"The file is larger than the {limit_mb} MB limit.")
    _check_content(kind, extension, data)
    return kind, MIME_TYPES[extension]
