"""Admin endpoints."""

import uuid
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select

from app.auth.deps import DbSession, require_role
from app.models import AgentTrace, User

router = APIRouter(prefix="/admin", tags=["admin"])

AdminUser = Annotated[User, Depends(require_role("admin"))]


class TraceStep(BaseModel):
    """One node's execution."""

    model_config = ConfigDict(from_attributes=True)

    step: int
    node: str
    input: dict[str, Any]
    output: dict[str, Any]
    latency_ms: int
    tokens_in: int
    tokens_out: int
    error: str | None
    created_at: datetime


class TraceOut(BaseModel):
    """Every step of one request."""

    trace_id: uuid.UUID
    steps: list[TraceStep]


@router.get("/traces/{trace_id}", response_model=TraceOut)
def get_trace(trace_id: uuid.UUID, db: DbSession, _: AdminUser) -> TraceOut:
    """Return what each node did for one request."""
    rows = db.scalars(
        select(AgentTrace).where(AgentTrace.trace_id == trace_id).order_by(AgentTrace.step)
    ).all()
    if not rows:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Trace not found")
    return TraceOut(trace_id=trace_id, steps=[TraceStep.model_validate(row) for row in rows])
