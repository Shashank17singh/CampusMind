"""FastAPI application factory."""

import redis
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.api import (
    admin,
    analytics,
    ask,
    auth,
    conversations,
    documents,
    reminders,
    study_plans,
    summaries,
    users,
    ws,
)
from app.api.errors import register_error_handling
from app.config import get_settings
from app.db import session_scope
from app.logging import configure_logging, get_logger
from app.metrics import install_metrics
from app.redis_client import get_redis

API_PREFIX = "/api/v1"

log = get_logger(__name__)


def _postgres_ok() -> bool:
    """Check the database."""
    try:
        with session_scope() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:
        log.warning("readiness_check_failed", dependency="postgres", error=str(exc))
        return False
    return True


def _redis_ok() -> bool:
    """Check Redis."""
    try:
        return bool(get_redis().ping())
    except redis.RedisError as exc:
        log.warning("readiness_check_failed", dependency="redis", error=str(exc))
        return False


def create_app() -> FastAPI:
    """Build the application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="CampusMind API", version="0.1.0")
    register_error_handling(app)
    install_metrics(app, get_redis())
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type"],
    )
    app.include_router(auth.router, prefix=API_PREFIX)
    app.include_router(users.router, prefix=API_PREFIX)
    app.include_router(documents.router, prefix=API_PREFIX)
    app.include_router(ask.router, prefix=API_PREFIX)
    app.include_router(conversations.router, prefix=API_PREFIX)
    app.include_router(admin.router, prefix=API_PREFIX)
    app.include_router(analytics.router, prefix=API_PREFIX)
    app.include_router(study_plans.router, prefix=API_PREFIX)
    app.include_router(reminders.router, prefix=API_PREFIX)
    app.include_router(summaries.router, prefix=API_PREFIX)
    app.include_router(ws.router, prefix=API_PREFIX)

    @app.get("/healthz", tags=["operations"])
    def healthz() -> dict[str, str]:
        """Report that the process is up."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["operations"])
    def readyz(response: Response) -> dict[str, object]:
        """Report whether the dependencies are reachable. Failure returns 503."""
        checks = {"postgres": _postgres_ok(), "redis": _redis_ok()}
        ready = all(checks.values())
        if not ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "ok" if ready else "unavailable", "checks": checks}

    return app


app = create_app()
