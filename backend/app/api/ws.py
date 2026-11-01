"""The notifications WebSocket.

Browsers cannot set an Authorization header on a WebSocket, and a token in the URL would
end up in access logs, so the client authenticates in its first message instead.
"""

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Annotated, Any, Protocol

import redis.asyncio
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from starlette.concurrency import run_in_threadpool

from app.auth.tokens import InvalidTokenError, decode_access_token
from app.config import get_settings
from app.db import session_scope
from app.logging import get_logger
from app.models import User

log = get_logger(__name__)

router = APIRouter()

AUTH_TIMEOUT_SECONDS = 5.0
CLOSE_UNAUTHORIZED = 4401


class NotificationBus(Protocol):
    """A stream of live messages for one user."""

    def subscribe(self, user_id: uuid.UUID) -> AsyncIterator[dict[str, Any]]:
        """Yield messages as they are published."""
        ...


class RedisNotificationBus:
    """Reads the Redis pub/sub channel that the scheduler publishes to."""

    def __init__(self, url: str) -> None:
        """Keep the Redis URL."""
        self._url = url

    async def subscribe(self, user_id: uuid.UUID) -> AsyncIterator[dict[str, Any]]:
        """Yield each message published for this user."""
        client = redis.asyncio.from_url(self._url, decode_responses=True)
        pubsub = client.pubsub()
        await pubsub.subscribe(f"notifications:{user_id}")
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    yield json.loads(message["data"])
        finally:
            await pubsub.aclose()
            await client.aclose()


def get_notification_bus() -> NotificationBus:
    """FastAPI dependency returning the bus."""
    return RedisNotificationBus(get_settings().redis_url)


def _authenticate(token: Any) -> uuid.UUID | None:
    """Return the id of the active user a token belongs to."""
    if not isinstance(token, str):
        return None
    try:
        user_id = uuid.UUID(decode_access_token(token)["sub"])
    except (InvalidTokenError, KeyError, ValueError):
        return None
    with session_scope() as session:
        user = session.get(User, user_id)
        return user_id if user is not None and user.is_active else None


async def _forward(websocket: WebSocket, messages: AsyncIterator[dict[str, Any]]) -> None:
    """Send each published message to the client."""
    async for message in messages:
        await websocket.send_json(message)


async def _answer_pings(websocket: WebSocket) -> None:
    """Reply to pings until the client disconnects."""
    while True:
        message = await websocket.receive_json()
        if isinstance(message, dict) and message.get("type") == "ping":
            await websocket.send_json({"type": "pong"})


async def _first_to_finish(*tasks: asyncio.Task[None]) -> None:
    """Wait for one task to end, then cancel the rest."""
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for task in done:
        # A disconnect ends the task with an exception, which is the normal way out.
        if not task.cancelled():
            task.exception()
    for task in pending:
        task.cancel()
        with suppress(asyncio.CancelledError, WebSocketDisconnect):
            await task


@router.websocket("/ws/notifications")
async def notifications(
    websocket: WebSocket,
    bus: Annotated[NotificationBus, Depends(get_notification_bus)],
) -> None:
    """Push a user's notifications to their browser."""
    await websocket.accept()
    try:
        first = await asyncio.wait_for(websocket.receive_json(), AUTH_TIMEOUT_SECONDS)
    except (TimeoutError, ValueError, WebSocketDisconnect):
        await _close_unauthorized(websocket)
        return

    token = first.get("token") if isinstance(first, dict) else None
    is_auth = isinstance(first, dict) and first.get("type") == "auth"
    user_id = await run_in_threadpool(_authenticate, token) if is_auth else None
    if user_id is None:
        await _close_unauthorized(websocket)
        return

    await websocket.send_json({"type": "ready"})
    await _first_to_finish(
        asyncio.create_task(_forward(websocket, bus.subscribe(user_id))),
        asyncio.create_task(_answer_pings(websocket)),
    )


async def _close_unauthorized(websocket: WebSocket) -> None:
    """Close with the application's unauthorized code."""
    with suppress(RuntimeError, WebSocketDisconnect):
        await websocket.close(code=CLOSE_UNAUTHORIZED)
