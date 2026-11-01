"""Signed tokens: JWT access tokens, and short-lived tokens sent by email."""

import hashlib
import secrets
import uuid
from datetime import timedelta
from typing import Any

import jwt

from app.config import get_settings
from app.models.base import utcnow

ALGORITHM = "HS256"
EMAIL_VERIFICATION_HOURS = 24
PASSWORD_RESET_MINUTES = 60


class InvalidTokenError(Exception):
    """The token is missing, malformed, expired or of the wrong type."""


def _encode(claims: dict[str, Any], lifetime: timedelta) -> str:
    """Sign claims with an issue time and an expiry."""
    now = utcnow()
    return jwt.encode(
        {**claims, "iat": now, "exp": now + lifetime},
        get_settings().secret_key,
        algorithm=ALGORITHM,
    )


def _decode(token: str, token_type: str) -> dict[str, Any]:
    """Verify a token's signature, expiry and type."""
    try:
        claims = jwt.decode(token, get_settings().secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError as exc:
        raise InvalidTokenError(str(exc)) from exc
    if claims.get("type") != token_type:
        raise InvalidTokenError(f"Not a {token_type} token")
    return claims


def create_access_token(user_id: uuid.UUID, role: str) -> str:
    """Issue a short-lived access token."""
    lifetime = timedelta(minutes=get_settings().access_token_minutes)
    return _encode({"sub": str(user_id), "role": role, "type": "access"}, lifetime)


def decode_access_token(token: str) -> dict[str, Any]:
    """Verify an access token and return its claims."""
    return _decode(token, "access")


def create_email_verification_token(user_id: uuid.UUID) -> str:
    """Issue a token proving ownership of an email address, valid for 24 hours."""
    lifetime = timedelta(hours=EMAIL_VERIFICATION_HOURS)
    return _encode({"sub": str(user_id), "type": "verify_email"}, lifetime)


def read_email_verification_token(token: str) -> uuid.UUID:
    """Return the user a verification token belongs to."""
    return uuid.UUID(_decode(token, "verify_email")["sub"])


def password_fingerprint(password_hash: str | None) -> str:
    """Summarise the current password. Changing it invalidates outstanding reset tokens."""
    return hashlib.sha256((password_hash or "").encode()).hexdigest()[:16]


def create_password_reset_token(user_id: uuid.UUID, password_hash: str | None) -> str:
    """Issue a reset token that stops working as soon as the password changes."""
    lifetime = timedelta(minutes=PASSWORD_RESET_MINUTES)
    claims = {
        "sub": str(user_id),
        "type": "reset_password",
        "pwd": password_fingerprint(password_hash),
    }
    return _encode(claims, lifetime)


def read_password_reset_token(token: str) -> tuple[uuid.UUID, str]:
    """Return the user and password fingerprint a reset token was issued for."""
    claims = _decode(token, "reset_password")
    return uuid.UUID(claims["sub"]), str(claims["pwd"])


def new_refresh_token() -> tuple[str, str]:
    """Create a random refresh token. Returns the token and the hash to store."""
    token = secrets.token_urlsafe(48)
    return token, hash_refresh_token(token)


def hash_refresh_token(token: str) -> str:
    """Hash a refresh token. It is already high-entropy, so a fast hash is enough."""
    return hashlib.sha256(token.encode()).hexdigest()
