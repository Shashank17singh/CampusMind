"""Password hashing with Argon2."""

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

_hasher = PasswordHasher()
# Verified when an account does not exist, so that case takes as long as a wrong password.
_DUMMY_HASH = _hasher.hash("campusmind-timing-equaliser")


def hash_password(password: str) -> str:
    """Hash a password."""
    return _hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    """Check a password in roughly constant time, even when there is no stored hash."""
    try:
        return _hasher.verify(password_hash or _DUMMY_HASH, password) and bool(password_hash)
    except (VerificationError, InvalidHashError):
        return False


def needs_rehash(password_hash: str) -> bool:
    """Return True if the hash was made with weaker parameters than the current ones."""
    return _hasher.check_needs_rehash(password_hash)
