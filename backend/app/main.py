"""FastAPI application factory."""

from fastapi import FastAPI
from sqlalchemy import text

from app.api import admin, ask, documents, reminders, study_plans, ws
from app.api.errors import register_error_handling
from app.config import get_settings
from app.db import session_scope
from app.logging import configure_logging, get_logger

API_PREFIX = "/api/v1"

log = get_logger(__name__)


def create_app() -> FastAPI:
    """Build the application."""
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(title="CampusMind API", version="0.1.0")
    register_error_handling(app)
    app.include_router(documents.router, prefix=API_PREFIX)
    app.include_router(ask.router, prefix=API_PREFIX)
    app.include_router(admin.router, prefix=API_PREFIX)
    app.include_router(study_plans.router, prefix=API_PREFIX)
    app.include_router(reminders.router, prefix=API_PREFIX)
    app.include_router(ws.router, prefix=API_PREFIX)

    @app.get("/healthz", tags=["operations"])
    def healthz() -> dict[str, str]:
        """Report that the process is up."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["operations"])
    def readyz() -> dict[str, object]:
        """Report whether the database is reachable."""
        try:
            with session_scope() as session:
                session.execute(text("SELECT 1"))
        except Exception as exc:
            log.warning("readiness_check_failed", error=str(exc))
            return {"status": "unavailable", "checks": {"postgres": False}}
        return {"status": "ok", "checks": {"postgres": True}}

    return app


app = create_app()
