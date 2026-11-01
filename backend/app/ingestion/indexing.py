"""The default indexer."""

import uuid
from collections.abc import Callable, Sequence

from app.ingestion.types import ChunkRecord
from app.logging import get_logger

log = get_logger(__name__)


class LoggingIndexer:
    """Accepts chunks without storing vectors. Replaced by the vector store."""

    def index(
        self, records: Sequence[ChunkRecord], on_progress: Callable[[int, int], None]
    ) -> None:
        """Report completion immediately."""
        log.warning("vector_store_not_configured", chunks=len(records))
        on_progress(len(records), len(records))

    def delete(self, document_id: uuid.UUID) -> None:
        """Nothing is stored, so nothing needs removing."""

    def mark_superseded(self, document_id: uuid.UUID) -> None:
        """Nothing is stored, so nothing needs flagging."""
