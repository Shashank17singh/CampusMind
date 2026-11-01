"""Writing the audit log."""

import ipaddress
import uuid
from typing import Any

from fastapi import Request
from sqlalchemy.orm import Session

from app.models import AuditLog


def _client_ip(request: Request | None) -> str | None:
    """Return the client address if it is a real IP. A bad value must not fail the action."""
    if request is None or request.client is None:
        return None
    try:
        return str(ipaddress.ip_address(request.client.host))
    except ValueError:
        return None


def record_audit(
    session: Session,
    action: str,
    *,
    actor_id: uuid.UUID | None = None,
    entity_type: str | None = None,
    entity_id: str | None = None,
    request: Request | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    """Add an audit entry to the current transaction, so it commits with the action."""
    session.add(
        AuditLog(
            actor_id=actor_id,
            action=action,
            entity_type=entity_type,
            entity_id=entity_id,
            ip_address=_client_ip(request),
            user_agent=request.headers.get("user-agent") if request else None,
            details=details or {},
        )
    )
