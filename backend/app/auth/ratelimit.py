"""Limiting failed logins per IP address and per account.

If Redis is down, this limiter stops limiting, but the account lockout in PostgreSQL still
applies. Users can keep logging in during an outage, and brute force remains bounded.
"""

import redis

from app.logging import get_logger

log = get_logger(__name__)

MAX_ATTEMPTS = 5
WINDOW_SECONDS = 15 * 60


class LoginLimiter:
    """A fixed-window counter of failed attempts."""

    def __init__(
        self, client: redis.Redis, limit: int = MAX_ATTEMPTS, window: int = WINDOW_SECONDS
    ) -> None:
        """Keep the Redis client and limits."""
        self._client = client
        self._limit = limit
        self._window = window

    @staticmethod
    def _keys(ip: str, email: str) -> tuple[str, str]:
        """Build the two counter keys."""
        return f"login:ip:{ip}", f"login:account:{email.lower()}"

    def is_blocked(self, ip: str, email: str) -> bool:
        """Return True if either counter has reached the limit."""
        try:
            values = self._client.mget(self._keys(ip, email))
        except redis.RedisError as exc:
            log.warning("login_limiter_unavailable", error=str(exc))
            return False
        return any(int(value or 0) >= self._limit for value in values)  # type: ignore[union-attr]

    def record_failure(self, ip: str, email: str) -> None:
        """Count a failed attempt. The first failure starts the window."""
        try:
            pipeline = self._client.pipeline()
            for key in self._keys(ip, email):
                pipeline.incr(key)
                pipeline.expire(key, self._window, nx=True)
            pipeline.execute()
        except redis.RedisError as exc:
            log.warning("login_limiter_unavailable", error=str(exc))

    def clear_account(self, email: str) -> None:
        """Forget an account's failures after a successful login."""
        try:
            self._client.delete(self._keys("", email)[1])
        except redis.RedisError as exc:
            log.warning("login_limiter_unavailable", error=str(exc))
