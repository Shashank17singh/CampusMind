"""Signed JWT access tokens."""

import uuid
from datetime import timedelta
from typing import Any

import jwt

from app.config import get_settings
from app.models.base import utcnow

ALGORITHM = "HS256"


class InvalidTokenError(Exception):
    """The token is missing, malformed, expired or of the wrong type."""


def create_access_token(user_id: uuid.UUID, role: str) -> str:
    """Issue a short-lived access token."""
    settings = get_settings()
    now = utcnow()
    claims = {
        "sub": str(user_id),
        "role": role,
        "type": "access",
        "iat": now,
        "exp": now + timedelta(minutes=settings.access_token_minutes),
    }
    return jwt.encode(claims, settings.secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify an access token and return its claims."""
    try:
        claims = jwt.decode(token, get_settings().secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc
    if claims.get("type") != "access":
        raise InvalidTokenError("Not an access token")
    return claims
