"""Registration, login, token rotation, password reset and bulk import."""

import csv
import io
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.auth.google import GoogleIdentity
from app.auth.passwords import hash_password, needs_rehash, verify_password
from app.auth.tokens import (
    InvalidTokenError,
    create_access_token,
    create_email_verification_token,
    create_password_reset_token,
    hash_refresh_token,
    new_refresh_token,
    password_fingerprint,
    read_email_verification_token,
    read_password_reset_token,
)
from app.config import get_settings
from app.email import EmailSender
from app.logging import get_logger
from app.models import RefreshToken, User
from app.models.base import utcnow

log = get_logger(__name__)

LOCKOUT_ATTEMPTS = 5
LOCKOUT_MINUTES = 15
CSV_REQUIRED_COLUMNS = {"email", "full_name"}


class AuthError(Exception):
    """An authentication failure. The message is safe to show to the user."""

    code = "unauthorized"
    status_code = 401


class EmailNotVerifiedError(AuthError):
    """The password was right, but the address has not been verified."""

    code = "email_not_verified"
    status_code = 403


class AccountLockedError(AuthError):
    """Too many failed attempts."""

    code = "account_locked"
    status_code = 429


class DomainNotAllowedError(AuthError):
    """The address is not on the college domain."""

    code = "email_domain_not_allowed"
    status_code = 422


class TokenReuseError(AuthError):
    """A spent refresh token was presented, which suggests it was stolen."""

    code = "token_reused"
    status_code = 401


@dataclass(frozen=True)
class TokenPair:
    """What a successful login or refresh hands back."""

    access_token: str
    refresh_token: str
    user: User


@dataclass
class ImportResult:
    """The outcome of a CSV import."""

    created: int = 0
    skipped: int = 0
    errors: list[str] = field(default_factory=list)


def normalize_email(email: str) -> str:
    """Addresses are compared in lowercase."""
    return email.strip().lower()


def domain_allowed(email: str) -> bool:
    """Check an address against the configured college domains."""
    allowed = get_settings().allowed_email_domains
    return email.rsplit("@", 1)[-1] in allowed


def find_user(session: Session, email: str) -> User | None:
    """Find a user by address, ignoring case."""
    return session.scalar(select(User).where(func.lower(User.email) == normalize_email(email)))


def _frontend_link(path: str, token: str) -> str:
    """Build a link into the web app."""
    return f"{get_settings().frontend_url.rstrip('/')}{path}?token={token}"


def send_verification(user: User, email: EmailSender) -> None:
    """Email a verification link."""
    token = create_email_verification_token(user.id)
    email.send(
        user.email,
        "Verify your CampusMind account",
        f"Open this link within 24 hours:\n{_frontend_link('/verify-email', token)}",
    )


def register(
    session: Session,
    email_sender: EmailSender,
    *,
    email: str,
    password: str,
    full_name: str,
    roll_number: str | None = None,
    department: str | None = None,
    semester: int | None = None,
    section: str | None = None,
) -> User | None:
    """Create a student account, or do nothing if the address is taken.

    The caller replies the same way in both cases, so this endpoint cannot be used to
    find out who has an account.
    """
    address = normalize_email(email)
    if not domain_allowed(address):
        raise DomainNotAllowedError("Please register with your college email address.")
    if find_user(session, address) is not None:
        log.info("registration_for_existing_account")
        return None
    user = User(
        email=address,
        password_hash=hash_password(password),
        full_name=full_name,
        roll_number=roll_number,
        department=department,
        semester=semester,
        section=section,
        role="student",
    )
    session.add(user)
    session.flush()
    send_verification(user, email_sender)
    return user


def verify_email(session: Session, token: str) -> None:
    """Mark an address as verified. Using a link twice does no harm."""
    try:
        user_id = read_email_verification_token(token)
    except (InvalidTokenError, ValueError, KeyError) as exc:
        raise AuthError("This verification link is invalid or has expired.") from exc
    user = session.get(User, user_id)
    if user is None:
        raise AuthError("This verification link is invalid or has expired.")
    user.email_verified_at = user.email_verified_at or utcnow()


def _issue(session: Session, user: User, family_id: uuid.UUID | None = None) -> TokenPair:
    """Create an access token and a refresh token in the given family."""
    token, token_hash = new_refresh_token()
    session.add(
        RefreshToken(
            user_id=user.id,
            family_id=family_id or uuid.uuid4(),
            token_hash=token_hash,
            expires_at=utcnow() + timedelta(days=get_settings().refresh_token_days),
        )
    )
    session.flush()
    return TokenPair(create_access_token(user.id, user.role), token, user)


def _record_failure(user: User, now: datetime) -> None:
    """Count a failed password, and lock the account at the limit."""
    user.failed_login_count += 1
    if user.failed_login_count >= LOCKOUT_ATTEMPTS:
        user.locked_until = now + timedelta(minutes=LOCKOUT_MINUTES)
        user.failed_login_count = 0
        log.warning("account_locked", user_id=str(user.id))


def login(session: Session, email: str, password: str) -> TokenPair:
    """Check credentials and start a new session family."""
    now = utcnow()
    user = find_user(session, email)
    if user is not None and user.locked_until and user.locked_until > now:
        raise AccountLockedError("Too many failed attempts. Try again in a few minutes.")

    valid = verify_password(user.password_hash if user else None, password)
    if user is None or not valid or not user.is_active:
        if user is not None:
            _record_failure(user, now)
        raise AuthError("Incorrect email or password.")
    if user.email_verified_at is None:
        raise EmailNotVerifiedError("Please verify your email address first.")

    user.failed_login_count = 0
    user.locked_until = None
    if user.password_hash and needs_rehash(user.password_hash):
        user.password_hash = hash_password(password)
    return _issue(session, user)


def google_login(session: Session, identity: GoogleIdentity) -> TokenPair:
    """Start a session for someone Google has vouched for, creating a student if new.

    An existing account with the same address is reused, whatever its role, and is marked
    verified because Google has verified the address.
    """
    address = normalize_email(identity.email)
    if not domain_allowed(address):
        raise DomainNotAllowedError("Please use your college Google account.")
    user = find_user(session, address)
    if user is None:
        user = User(email=address, full_name=identity.name, role="student")
        session.add(user)
        session.flush()
        log.info("google_account_created", user_id=str(user.id))
    if not user.is_active:
        raise AuthError("This account has been deactivated.")
    user.email_verified_at = user.email_verified_at or utcnow()
    return _issue(session, user)


def revoke_family(session: Session, family_id: uuid.UUID) -> None:
    """End every token in a session family."""
    session.execute(
        update(RefreshToken)
        .where(RefreshToken.family_id == family_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )


def rotate(session: Session, token: str) -> TokenPair:
    """Swap a refresh token for a new one.

    Each token works once. If a spent token comes back, then either the real user or a
    thief is holding a stale copy, and there is no way to tell which, so the whole family
    is revoked and the real user simply has to log in again. The row lock means two
    simultaneous requests cannot both spend one token.
    """
    row = session.scalar(
        select(RefreshToken)
        .where(RefreshToken.token_hash == hash_refresh_token(token))
        .with_for_update()
    )
    now = utcnow()
    if row is None or row.revoked_at is not None or row.expires_at <= now:
        raise AuthError("Your session has expired. Please log in again.")
    if row.used_at is not None:
        log.warning("refresh_token_reuse_detected", family_id=str(row.family_id))
        revoke_family(session, row.family_id)
        raise TokenReuseError("Your session has expired. Please log in again.")

    user = session.get(User, row.user_id)
    if user is None or not user.is_active:
        raise AuthError("Your session has expired. Please log in again.")
    row.used_at = now
    return _issue(session, user, row.family_id)


def logout(session: Session, token: str | None) -> None:
    """End the session the refresh token belongs to."""
    if token is None:
        return
    row = session.scalar(
        select(RefreshToken).where(RefreshToken.token_hash == hash_refresh_token(token))
    )
    if row is not None:
        revoke_family(session, row.family_id)


def request_password_reset(session: Session, email_sender: EmailSender, email: str) -> None:
    """Email a reset link if the account exists. The caller's reply never says either way."""
    user = find_user(session, email)
    if user is None or not user.is_active:
        return
    token = create_password_reset_token(user.id, user.password_hash)
    email_sender.send(
        user.email,
        "Reset your CampusMind password",
        f"Open this link within an hour:\n{_frontend_link('/reset-password', token)}",
    )


def reset_password(session: Session, token: str, new_password: str) -> User:
    """Set a new password and sign out every existing session."""
    invalid = AuthError("This reset link is invalid or has expired.")
    try:
        user_id, fingerprint = read_password_reset_token(token)
    except (InvalidTokenError, ValueError, KeyError) as exc:
        raise invalid from exc
    user = session.get(User, user_id)
    if user is None or fingerprint != password_fingerprint(user.password_hash):
        raise invalid

    user.password_hash = hash_password(new_password)
    user.failed_login_count = 0
    user.locked_until = None
    # A link sent to this address proves ownership of it.
    user.email_verified_at = user.email_verified_at or utcnow()
    session.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user.id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=utcnow())
    )
    return user


def _parse_semester(value: str | None) -> int | None:
    """Read a semester cell, rejecting anything outside 1 to 8."""
    text = (value or "").strip()
    if not text:
        return None
    semester = int(text)
    if not 1 <= semester <= 8:
        raise ValueError("semester must be between 1 and 8")
    return semester


def import_students(session: Session, content: bytes) -> ImportResult:
    """Create student accounts from a CSV. Each row stands alone, so one bad row stops nothing.

    Imported students have no password. They set one with the forgot-password flow, which
    also proves that they own the address.
    """
    result = ImportResult()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        result.errors.append("The file is not UTF-8 text.")
        return result

    reader = csv.DictReader(io.StringIO(text))
    columns = {name.strip().lower() for name in reader.fieldnames or []}
    missing = CSV_REQUIRED_COLUMNS - columns
    if missing:
        result.errors.append(f"Missing columns: {', '.join(sorted(missing))}")
        return result

    for number, raw in enumerate(reader, start=2):
        row = {(key or "").strip().lower(): (value or "").strip() for key, value in raw.items()}
        email = normalize_email(row.get("email", ""))
        if "@" not in email or not row.get("full_name"):
            result.errors.append(f"Row {number}: email and full_name are required")
            continue
        if not domain_allowed(email):
            result.errors.append(f"Row {number}: {email} is not a college address")
            continue
        try:
            semester = _parse_semester(row.get("semester"))
        except ValueError as exc:
            result.errors.append(f"Row {number}: {exc}")
            continue

        roll_number = row.get("roll_number") or None
        taken = find_user(session, email) is not None or (
            roll_number is not None
            and session.scalar(select(User.id).where(User.roll_number == roll_number)) is not None
        )
        if taken:
            result.skipped += 1
            continue
        session.add(
            User(
                email=email,
                full_name=row["full_name"],
                roll_number=roll_number,
                department=row.get("department") or None,
                semester=semester,
                section=row.get("section") or None,
                role="student",
            )
        )
        session.flush()
        result.created += 1
    return result
