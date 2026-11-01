"""Prometheus metrics for requests and the Celery queue."""

import time
from collections.abc import Awaitable, Callable, Iterable

import redis
from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Histogram,
    generate_latest,
)
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

from app.logging import get_logger

log = get_logger(__name__)

QUEUE_NAME = "celery"
LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60)


class QueueCollector(Collector):
    """Reads the Celery queue length at scrape time."""

    def __init__(self, client: redis.Redis) -> None:
        """Keep the Redis client."""
        self._client = client

    def collect(self) -> Iterable[GaugeMetricFamily]:
        """Report the queue length, or nothing if Redis cannot be reached."""
        try:
            length = float(self._client.llen(QUEUE_NAME))  # type: ignore[arg-type]
        except redis.RedisError as exc:
            log.warning("queue_metric_unavailable", error=str(exc))
            return
        yield GaugeMetricFamily("campusmind_queue_length", "Tasks waiting in Celery", value=length)


def _route_label(request: Request) -> str:
    """Use the route template, so ids cannot create unlimited label values."""
    route = request.scope.get("route")
    return getattr(route, "path", "unmatched")


def install_metrics(app: FastAPI, client: redis.Redis | None = None) -> None:
    """Record every request, and serve /metrics. Nginx blocks that path from the internet."""
    registry = CollectorRegistry()
    requests = Counter(
        "campusmind_requests",
        "HTTP requests",
        ["method", "route", "status"],
        registry=registry,
    )
    latency = Histogram(
        "campusmind_request_seconds",
        "HTTP request latency",
        ["method", "route"],
        buckets=LATENCY_BUCKETS,
        registry=registry,
    )
    if client is not None:
        registry.register(QueueCollector(client))

    @app.middleware("http")
    async def measure(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = _route_label(request)
            requests.labels(request.method, route, str(status)).inc()
            latency.labels(request.method, route).observe(time.perf_counter() - started)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Expose metrics in Prometheus format."""
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)
