"""Celery tasks for reminders."""

from app.celery_app import celery_app
from app.db import get_session_factory
from app.email import get_email_sender
from app.models.base import utcnow
from app.redis_client import get_redis
from app.reminders.scheduler import RedisPublisher, dispatch_due


@celery_app.task(name="reminders.dispatch", ignore_result=True)
def dispatch() -> int:
    """Run one scheduler tick. Celery beat calls this every minute."""
    return dispatch_due(
        get_session_factory(), utcnow(), get_email_sender(), RedisPublisher(get_redis())
    )
