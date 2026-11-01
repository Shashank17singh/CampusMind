"""Authentication endpoints."""

from typing import Annotated

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr, Field

from app.api.users import UserOut
from app.audit import record_audit
from app.auth import service
from app.auth.deps import DbSession
from app.auth.ratelimit import LoginLimiter
from app.auth.service import AuthError, TokenReuseError
from app.config import get_settings
from app.email import EmailSender, get_email_sender
from app.logging import get_logger
from app.redis_client import get_redis

router = APIRouter(prefix="/auth", tags=["auth"])
log = get_logger(__name__)

COOKIE_NAME = "campusmind_refresh"
COOKIE_PATH = "/api/v1/auth"
ACCEPTED_MESSAGE = "If that address is valid, a message is on its way."

Mailer = Annotated[EmailSender, Depends(get_email_sender)]
RefreshCookie = Annotated[str | None, Cookie(alias=COOKIE_NAME)]


def get_login_limiter() -> LoginLimiter:
    """FastAPI dependency returning the login limiter."""
    return LoginLimiter(get_redis())


Limiter = Annotated[LoginLimiter, Depends(get_login_limiter)]

Password = Annotated[str, Field(min_length=10, max_length=128)]


class RegisterRequest(BaseModel):
    """A new account."""

    email: EmailStr
    password: Password
    full_name: str = Field(min_length=1, max_length=200)
    roll_number: str | None = Field(default=None, max_length=50)
    department: str | None = Field(default=None, max_length=50)
    semester: int | None = Field(default=None, ge=1, le=8)
    section: str | None = Field(default=None, max_length=20)


class MessageOut(BaseModel):
    """A plain confirmation."""

    message: str


class TokenRequest(BaseModel):
    """A token from an email link."""

    token: str


class EmailRequest(BaseModel):
    """An email address."""

    email: EmailStr


class LoginRequest(BaseModel):
    """Login credentials."""

    email: EmailStr
    password: str = Field(max_length=128)


class ResetRequest(BaseModel):
    """A new password, with the token that authorises it."""

    token: str
    new_password: Password


class LoginOut(BaseModel):
    """A new session."""

    access_token: str
    expires_in: int
    user: UserOut


class AccessOut(BaseModel):
    """A new access token."""

    access_token: str
    expires_in: int


def _raise(error: AuthError) -> HTTPException:
    """Convert an auth failure to an HTTP error."""
    return HTTPException(error.status_code, detail=str(error))


def _client_ip(request: Request) -> str:
    """Return the caller's address for rate limiting."""
    return request.client.host if request.client else "unknown"


def _set_cookie(response: Response, token: str) -> None:
    """Store the refresh token where page scripts cannot read it."""
    settings = get_settings()
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=settings.refresh_token_days * 86400,
        httponly=True,
        secure=settings.cookie_secure,
        samesite="strict",
        path=COOKIE_PATH,
    )


def _access_lifetime() -> int:
    """Return the access token's lifetime in seconds."""
    return get_settings().access_token_minutes * 60


@router.post("/register", response_model=MessageOut, status_code=status.HTTP_202_ACCEPTED)
def register(body: RegisterRequest, db: DbSession, mailer: Mailer) -> MessageOut:
    """Create a student account. The reply is the same if the address is already registered."""
    try:
        service.register(
            db,
            mailer,
            email=body.email,
            password=body.password,
            full_name=body.full_name,
            roll_number=body.roll_number,
            department=body.department,
            semester=body.semester,
            section=body.section,
        )
    except AuthError as exc:
        raise _raise(exc) from exc
    db.commit()
    return MessageOut(message=ACCEPTED_MESSAGE)


@router.post("/verify-email", response_model=MessageOut)
def verify_email(body: TokenRequest, db: DbSession) -> MessageOut:
    """Confirm an email address."""
    try:
        service.verify_email(db, body.token)
    except AuthError as exc:
        raise _raise(exc) from exc
    db.commit()
    return MessageOut(message="Your email address is verified.")


@router.post("/resend-verification", response_model=MessageOut, status_code=202)
def resend_verification(body: EmailRequest, db: DbSession, mailer: Mailer) -> MessageOut:
    """Send another verification link."""
    user = service.find_user(db, body.email)
    if user is not None and user.email_verified_at is None:
        service.send_verification(user, mailer)
    return MessageOut(message=ACCEPTED_MESSAGE)


@router.post("/login", response_model=LoginOut)
def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: DbSession,
    limiter: Limiter,
) -> LoginOut:
    """Log in, returning an access token and setting the refresh cookie."""
    ip = _client_ip(request)
    if limiter.is_blocked(ip, body.email):
        record_audit(
            db, "auth.login_blocked", request=request, details={"email": body.email.lower()}
        )
        db.commit()
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many attempts. Try again in a few minutes.",
        )
    try:
        pair = service.login(db, body.email, body.password)
    except AuthError as exc:
        limiter.record_failure(ip, body.email)
        record_audit(
            db,
            "auth.login_failed",
            request=request,
            details={"email": body.email.lower(), "reason": exc.code},
        )
        # Commit even though this request fails, so the failure count and audit row stay.
        db.commit()
        raise _raise(exc) from exc

    limiter.clear_account(body.email)
    record_audit(db, "auth.login", actor_id=pair.user.id, entity_type="user", request=request)
    db.commit()
    _set_cookie(response, pair.refresh_token)
    return LoginOut(
        access_token=pair.access_token,
        expires_in=_access_lifetime(),
        user=UserOut.model_validate(pair.user),
    )


@router.post("/refresh", response_model=AccessOut)
def refresh(
    request: Request,
    response: Response,
    db: DbSession,
    token: RefreshCookie = None,
) -> AccessOut:
    """Swap the refresh cookie for a new one and a new access token."""
    if token is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        pair = service.rotate(db, token)
    except TokenReuseError as exc:
        record_audit(db, "auth.refresh_reuse", request=request)
        # The family revocation must survive this failed request.
        db.commit()
        response.delete_cookie(COOKIE_NAME, path=COOKIE_PATH)
        raise _raise(exc) from exc
    except AuthError as exc:
        raise _raise(exc) from exc
    db.commit()
    _set_cookie(response, pair.refresh_token)
    return AccessOut(access_token=pair.access_token, expires_in=_access_lifetime())


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(
    request: Request, response: Response, db: DbSession, token: RefreshCookie = None
) -> None:
    """End the session. This works with an expired access token too."""
    service.logout(db, token)
    record_audit(db, "auth.logout", request=request)
    db.commit()
    response.delete_cookie(COOKIE_NAME, path=COOKIE_PATH)


@router.post("/forgot-password", response_model=MessageOut, status_code=202)
def forgot_password(body: EmailRequest, db: DbSession, mailer: Mailer) -> MessageOut:
    """Send a reset link. The reply never says whether the account exists."""
    service.request_password_reset(db, mailer, body.email)
    return MessageOut(message=ACCEPTED_MESSAGE)


@router.post("/reset-password", status_code=status.HTTP_204_NO_CONTENT)
def reset_password(body: ResetRequest, request: Request, db: DbSession) -> None:
    """Set a new password and end every existing session."""
    try:
        user = service.reset_password(db, body.token, body.new_password)
    except AuthError as exc:
        raise _raise(exc) from exc
    record_audit(db, "auth.password_reset", actor_id=user.id, request=request)
    db.commit()
