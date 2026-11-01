"""Celery tasks for conversations."""

import uuid

from app.celery_app import celery_app
from app.config import get_settings
from app.conversations.service import delete_expired, make_title
from app.db import session_scope
from app.llm import get_llm
from app.logging import get_logger
from app.models import Conversation

log = get_logger(__name__)


@celery_app.task(name="conversations.auto_title", ignore_result=True)
def auto_title(conversation_id: str, question: str) -> None:
    """Title a new conversation, unless the user has named it in the meantime."""
    title = make_title(question, get_llm())
    with session_scope() as session:
        conversation = session.get(Conversation, uuid.UUID(conversation_id))
        if conversation is not None and conversation.title is None:
            conversation.title = title


@celery_app.task(name="conversations.delete_expired", ignore_result=True)
def delete_expired_conversations() -> int:
    """Apply the retention policy. Celery beat runs this daily."""
    with session_scope() as session:
        deleted = delete_expired(session, get_settings().conversation_retention_days)
    if deleted:
        log.info("conversations_deleted", count=deleted)
    return deleted
