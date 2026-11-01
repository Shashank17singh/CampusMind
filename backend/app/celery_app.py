"""Celery application."""

from celery import Celery

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "campusmind",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.ingestion.tasks"],
)
celery_app.conf.update(
    task_always_eager=settings.celery_eager,
    task_eager_propagates=settings.celery_eager,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_ignore_result=True,
    timezone="UTC",
    worker_prefetch_multiplier=1,
)
