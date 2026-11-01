"""The shared Redis client."""

from functools import lru_cache

import redis

from app.config import get_settings


@lru_cache
def get_redis() -> redis.Redis:
    """Return the process-wide client. Replies are decoded to str."""
    return redis.Redis.from_url(
        get_settings().redis_url, decode_responses=True, socket_timeout=2, socket_connect_timeout=2
    )
