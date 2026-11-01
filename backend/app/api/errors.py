"""One error shape for the whole API, and a trace id on every request."""

import uuid
from collections.abc import Awaitable, Callable, Mapping

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.ingestion.errors import IngestionError
from app.logging import get_logger

log = get_logger(__name__)


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    """Build the standard error body."""
    trace_id = getattr(request.state, "trace_id", None)
    body = {"error": {"code": code, "message": message, "trace_id": trace_id}}
    return JSONResponse(status_code=status_code, content=body, headers=headers)


def register_error_handling(app: FastAPI) -> None:
    """Install the trace id middleware and the exception handlers."""

    @app.middleware("http")
    async def trace_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        trace_id = str(uuid.uuid4())
        request.state.trace_id = trace_id
        structlog.contextvars.bind_contextvars(trace_id=trace_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Trace-Id"] = trace_id
        return response

    @app.exception_handler(IngestionError)
    async def on_ingestion_error(request: Request, exc: IngestionError) -> JSONResponse:
        return _error_response(request, exc.status_code, exc.code, str(exc))

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: "unauthorized", 403: "forbidden", 404: "not_found"}.get(
            exc.status_code, "http_error"
        )
        return _error_response(request, exc.status_code, code, str(exc.detail), exc.headers)

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(part) for part in first.get("loc", ()))
        message = f"{where}: {first.get('msg', 'invalid request')}"
        return _error_response(request, 422, "validation_error", message)

    @app.exception_handler(Exception)
    async def on_unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        log.error("unhandled_exception", path=request.url.path, exc_info=exc)
        return _error_response(request, 500, "internal_error", "Something went wrong.")
