"""Stateless question answering over Server-Sent Events."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.auth.deps import CurrentUser
from app.llm import ChatMessage
from app.rag.pipeline import HISTORY_MESSAGES, RagPipeline
from app.rag.schemas import RagRequest, UserProfile
from app.rag.service import get_rag_pipeline, sse_stream

router = APIRouter(tags=["chat"])


class HistoryMessage(BaseModel):
    """One earlier message."""

    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class AskRequest(BaseModel):
    """A question, with any earlier conversation."""

    question: str = Field(min_length=1, max_length=2000)
    history: list[HistoryMessage] = Field(default_factory=list)


@router.post("/ask")
def ask(
    body: AskRequest,
    request: Request,
    user: CurrentUser,
    pipeline: Annotated[RagPipeline, Depends(get_rag_pipeline)],
) -> StreamingResponse:
    """Answer one question, streaming tokens and then its citations."""
    rag_request = RagRequest(
        question=body.question,
        profile=UserProfile(department=user.department, semester=user.semester),
        history=tuple(
            ChatMessage(item.role, item.content) for item in body.history[-HISTORY_MESSAGES:]
        ),
    )
    return StreamingResponse(
        sse_stream(pipeline, rag_request, request.state.trace_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
