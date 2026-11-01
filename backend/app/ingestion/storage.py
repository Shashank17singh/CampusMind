"""Local file storage for uploads."""

import os
import tempfile
from pathlib import Path


class LocalStorage:
    """Stores uploads under a root folder, named by content hash.

    The uploader's filename is never used in the path, which rules out path traversal.
    """

    def __init__(self, root: Path) -> None:
        """Create the storage root if needed."""
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def save(self, sha256: str, extension: str, data: bytes) -> str:
        """Write a file atomically and return its path relative to the root."""
        name = f"{sha256}{extension.lower()}"
        file_descriptor, temp_path = tempfile.mkstemp(dir=self._root)
        try:
            with os.fdopen(file_descriptor, "wb") as handle:
                handle.write(data)
            os.replace(temp_path, self._root / name)
        except OSError:
            Path(temp_path).unlink(missing_ok=True)
            raise
        return name

    def path(self, relative: str) -> Path:
        """Return the absolute path of a stored file."""
        return self._root / relative

    def delete(self, relative: str) -> None:
        """Remove a stored file if it exists."""
        self.path(relative).unlink(missing_ok=True)
