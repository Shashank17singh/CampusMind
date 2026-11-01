"""Profile and user management endpoints."""

import uuid
from datetime import datetime
from typing import Annotated, Literal
from zoneinfo import available_timezones

from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import ColumnElement, func, or_, select

from app.audit import record_audit
from app.auth.deps import CurrentUser, DbSession, require_role
from app.auth.service import import_students
from app.models import User

router = APIRouter(tags=["users"])

AdminUser = Annotated[User, Depends(require_role("admin"))]

MAX_CSV_BYTES = 2 * 1024 * 1024


class UserOut(BaseModel):
    """A user, without anything secret."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    email: str
    full_name: str
    roll_number: str | None
    role: str
    department: str | None
    semester: int | None
    section: str | None
    timezone: str
    is_active: bool
    email_verified_at: datetime | None
    created_at: datetime


class ProfileUpdate(BaseModel):
    """The fields users can change about themselves. Role is deliberately absent."""

    full_name: str | None = Field(default=None, min_length=1, max_length=200)
    department: str | None = Field(default=None, max_length=50)
    semester: int | None = Field(default=None, ge=1, le=8)
    section: str | None = Field(default=None, max_length=20)
    timezone: str | None = None

    @field_validator("timezone")
    @classmethod
    def _check_timezone(cls, value: str | None) -> str | None:
        """Only real timezone names can be stored."""
        if value is not None and value not in available_timezones():
            raise ValueError("Unknown timezone")
        return value


class AdminUserUpdate(BaseModel):
    """The fields an admin can change."""

    role: Literal["student", "faculty", "admin"] | None = None
    is_active: bool | None = None


class UserPage(BaseModel):
    """One page of users."""

    items: list[UserOut]
    total: int


class ImportOut(BaseModel):
    """The outcome of a CSV import."""

    created: int
    skipped: int
    errors: list[str]


@router.get("/users/me", response_model=UserOut)
def get_me(user: CurrentUser) -> User:
    """Return the signed-in user."""
    return user


@router.patch("/users/me", response_model=UserOut)
def update_me(body: ProfileUpdate, db: DbSession, user: CurrentUser) -> User:
    """Update the signed-in user's profile."""
    for key, value in body.model_dump(exclude_unset=True).items():
        setattr(user, key, value)
    db.commit()
    return user


@router.get("/admin/users", response_model=UserPage)
def list_users(
    db: DbSession,
    _: AdminUser,
    role: str | None = None,
    department: str | None = None,
    q: Annotated[str | None, Query(max_length=100)] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> UserPage:
    """Search users."""
    filters: list[ColumnElement[bool]] = []
    if role:
        filters.append(User.role == role)
    if department:
        filters.append(User.department == department)
    if q:
        # A literal % or _ in the search box must not act as a wildcard.
        literal = q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{literal}%"
        filters.append(
            or_(
                User.email.ilike(pattern, escape="\\"),
                User.full_name.ilike(pattern, escape="\\"),
            )
        )
    total = db.scalar(select(func.count()).select_from(User).where(*filters)) or 0
    rows = db.scalars(
        select(User).where(*filters).order_by(User.created_at.desc()).limit(limit).offset(offset)
    ).all()
    return UserPage(items=[UserOut.model_validate(row) for row in rows], total=total)


@router.patch("/admin/users/{user_id}", response_model=UserOut)
def update_user(
    user_id: uuid.UUID,
    body: AdminUserUpdate,
    request: Request,
    db: DbSession,
    admin: AdminUser,
) -> User:
    """Change a user's role or active state. Admins cannot lock themselves out."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="User not found")
    if user.id == admin.id and (body.role not in (None, "admin") or body.is_active is False):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, detail="You cannot remove your own admin access"
        )
    changes = body.model_dump(exclude_unset=True, exclude_none=True)
    for key, value in changes.items():
        setattr(user, key, value)
    record_audit(
        db,
        "user.update",
        actor_id=admin.id,
        entity_type="user",
        entity_id=str(user.id),
        request=request,
        details=changes,
    )
    db.commit()
    return user


@router.post("/admin/users/import", response_model=ImportOut)
def import_users(
    request: Request,
    db: DbSession,
    admin: AdminUser,
    file: Annotated[UploadFile, File()],
) -> ImportOut:
    """Bulk-create students from a CSV with email and full_name columns."""
    content = file.file.read(MAX_CSV_BYTES + 1)
    if len(content) > MAX_CSV_BYTES:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE, detail="The CSV is larger than the 2 MB limit"
        )
    result = import_students(db, content)
    record_audit(
        db,
        "user.import",
        actor_id=admin.id,
        request=request,
        details={"created": result.created, "skipped": result.skipped},
    )
    db.commit()
    return ImportOut(created=result.created, skipped=result.skipped, errors=result.errors)
