"""Builds the production agent graph with every real agent attached."""

import uuid
from functools import lru_cache

from app.agents.graph import AgentRunner
from app.agents.planner import make_planner_handler
from app.agents.qa import make_qa_handler
from app.agents.summarizer import make_summarizer_handler
from app.agents.summary_jobs import run_summary_task
from app.agents.tracing import TraceRecorder
from app.config import get_settings
from app.db import get_session_factory, session_scope
from app.llm import get_llm
from app.models import Document, Summary
from app.rag.service import get_rag_pipeline
from app.reminders.service import make_reminder_handler
from app.vectorstore import get_retriever


def _queue_summary(document_id: uuid.UUID, mode: str, user_id: uuid.UUID) -> None:
    """Create a summary row and hand it to the background worker."""
    with session_scope() as session:
        document = session.get_one(Document, document_id)
        summary = Summary(
            user_id=user_id, document_id=document_id, source_name=document.title, mode=mode
        )
        session.add(summary)
        session.flush()
        summary_id = summary.id
    run_summary_task.delay(str(summary_id))


@lru_cache
def get_agent_runner() -> AgentRunner:
    """Return the process-wide runner, built on first use."""
    llm = get_llm()
    sessions = get_session_factory()
    return AgentRunner(
        llm,
        {
            "qa": make_qa_handler(get_rag_pipeline()),
            "study_plan": make_planner_handler(llm, sessions),
            "reminder": make_reminder_handler(llm, sessions),
            "summarize": make_summarizer_handler(
                llm,
                get_retriever(),
                sessions,
                _queue_summary,
                get_settings().llm_max_concurrency,
            ),
        },
        TraceRecorder(sessions),
    )
