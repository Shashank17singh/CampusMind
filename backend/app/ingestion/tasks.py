"""Celery tasks for ingestion."""

import uuid

from sqlalchemy.exc import OperationalError

from app.celery_app import celery_app
from app.config import get_settings
from app.db import get_session_factory
from app.ingestion.ocr import TesseractOcr
from app.ingestion.pipeline import PipelineDeps, run_ingestion
from app.ingestion.storage import LocalStorage
from app.llm import get_llm
from app.rag.service import get_answer_cache
from app.vectorstore import get_vector_store
from app.vectorstore.embeddings import get_embedder
from app.vectorstore.indexer import VectorIndexer


def build_deps() -> PipelineDeps:
    """Assemble the real pipeline dependencies."""
    settings = get_settings()
    return PipelineDeps(
        session_factory=get_session_factory(),
        storage=LocalStorage(settings.upload_dir),
        ocr=TesseractOcr(settings.tesseract_cmd),
        llm=get_llm(),
        indexer=VectorIndexer(get_vector_store(), get_embedder()),
        settings=settings,
        on_ready=lambda _: get_answer_cache().invalidate(),
    )


@celery_app.task(
    bind=True,
    name="ingestion.ingest_document",
    acks_late=True,
    autoretry_for=(OperationalError,),
    retry_backoff=True,
    max_retries=3,
    soft_time_limit=600,
    time_limit=660,
)
def ingest_document(self, document_id: str) -> bool:  # type: ignore[no-untyped-def]
    """Run the pipeline for one document."""
    return run_ingestion(build_deps(), uuid.UUID(document_id), force=self.request.retries > 0)
