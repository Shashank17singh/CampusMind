"""Study plan endpoints."""

import uuid
from datetime import date, datetime, time
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select

from app.agents.planner import (
    PlanInputs,
    PlannerError,
    create_plan,
    export_ics,
    get_user_plan,
    replan,
)
from app.agents.state import UserContext
from app.auth.deps import CurrentUser, DbSession
from app.llm import LLMClient, get_llm
from app.models import PlanItem, StudyPlan

router = APIRouter(prefix="/study-plans", tags=["study plans"])


class PlanItemOut(BaseModel):
    """One scheduled slot."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    date: date
    start_time: time
    end_time: time
    subject: str
    topic: str
    type: str
    status: str


class StudyPlanOut(BaseModel):
    """A plan with its items."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    start_date: date
    end_date: date
    is_active: bool
    created_at: datetime
    items: list[PlanItemOut]
    warnings: list[str] = []


class PlanSummary(BaseModel):
    """A plan without its items."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    start_date: date
    end_date: date
    is_active: bool
    created_at: datetime


class PlanPage(BaseModel):
    """One page of plans."""

    items: list[PlanSummary]
    total: int


class ItemUpdate(BaseModel):
    """A status change for one slot."""

    status: Literal["pending", "done", "missed"]


class ReplanRequest(BaseModel):
    """When to restart the schedule from. Defaults to today."""

    from_date: date | None = None


def _user_context(user: CurrentUser) -> UserContext:
    """Convert a user row to the agent layer's view of it."""
    return UserContext(user.id, user.role, user.department, user.semester, user.timezone)


def _load(db: DbSession, plan_id: uuid.UUID, user: CurrentUser) -> StudyPlan:
    """Load the user's own plan. Someone else's plan looks the same as a missing one."""
    plan = get_user_plan(db, plan_id, user.id)
    if plan is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Study plan not found")
    return plan


def _out(plan: StudyPlan, warnings: list[str] | None = None) -> StudyPlanOut:
    """Build the response for a plan."""
    result = StudyPlanOut.model_validate(plan)
    result.warnings = warnings or []
    return result


@router.post("", response_model=StudyPlanOut, status_code=status.HTTP_201_CREATED)
def create_study_plan(
    body: PlanInputs,
    db: DbSession,
    user: CurrentUser,
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> StudyPlanOut:
    """Build a plan from exam dates, available hours and difficulty ratings."""
    try:
        plan, _, warnings = create_plan(db, llm, _user_context(user), body)
    except PlannerError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)) from exc
    db.commit()
    db.refresh(plan, ["items"])
    return _out(plan, warnings)


@router.get("", response_model=PlanPage)
def list_plans(
    db: DbSession,
    user: CurrentUser,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> PlanPage:
    """List the user's plans, newest first."""
    total = db.scalar(
        select(func.count()).select_from(StudyPlan).where(StudyPlan.user_id == user.id)
    )
    rows = db.scalars(
        select(StudyPlan)
        .where(StudyPlan.user_id == user.id)
        .order_by(StudyPlan.created_at.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return PlanPage(items=[PlanSummary.model_validate(row) for row in rows], total=total or 0)


@router.get("/{plan_id}", response_model=StudyPlanOut)
def get_plan(plan_id: uuid.UUID, db: DbSession, user: CurrentUser) -> StudyPlanOut:
    """Return a plan with its items."""
    return _out(_load(db, plan_id, user))


@router.patch("/{plan_id}/items/{item_id}", response_model=PlanItemOut)
def update_item(
    plan_id: uuid.UUID,
    item_id: uuid.UUID,
    body: ItemUpdate,
    db: DbSession,
    user: CurrentUser,
) -> PlanItem:
    """Mark a slot done or missed."""
    _load(db, plan_id, user)
    item = db.get(PlanItem, item_id)
    if item is None or item.plan_id != plan_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Plan item not found")
    item.status = body.status
    db.commit()
    return item


@router.post("/{plan_id}/replan", response_model=StudyPlanOut)
def replan_plan(
    plan_id: uuid.UUID, body: ReplanRequest, db: DbSession, user: CurrentUser
) -> StudyPlanOut:
    """Redistribute unfinished work without moving completed items."""
    plan = _load(db, plan_id, user)
    result = replan(db, plan, body.from_date)
    db.commit()
    db.expire(plan, ["items"])
    warnings = [f"{topic.name} no longer fits before its exam." for topic in result.unscheduled[:5]]
    return _out(plan, warnings)


@router.get("/{plan_id}/export.ics")
def export_plan(plan_id: uuid.UUID, db: DbSession, user: CurrentUser) -> Response:
    """Download the plan as a calendar file."""
    plan = _load(db, plan_id, user)
    return Response(
        content=export_ics(db, plan, user.timezone),
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": 'attachment; filename="study-plan.ics"'},
    )


@router.delete("/{plan_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_plan(plan_id: uuid.UUID, db: DbSession, user: CurrentUser) -> None:
    """Delete a plan and its items."""
    db.delete(_load(db, plan_id, user))
    db.commit()
