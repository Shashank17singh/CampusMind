"""Tests for metrics, readiness, the sample data and the infrastructure files."""

import json
from pathlib import Path

import fakeredis
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.main as main_module
from app.metrics import install_metrics
from scripts.sample_documents import BANNER, SAMPLES

ROOT = Path(__file__).resolve().parents[2]


def metrics_app(client: object = None) -> TestClient:
    """Build a small app with metrics installed."""
    app = FastAPI()
    install_metrics(app, client)  # type: ignore[arg-type]

    @app.get("/items/{item_id}")
    def item(item_id: str) -> dict[str, str]:
        return {"id": item_id}

    return TestClient(app)


class TestMetrics:
    def test_requests_are_counted_by_route_template(self) -> None:
        client = metrics_app()
        client.get("/items/1")
        client.get("/items/2")
        body = client.get("/metrics").text
        assert 'route="/items/{item_id}"' in body
        assert 'status="200"' in body
        assert "/items/1" not in body

    def test_latency_is_recorded(self) -> None:
        client = metrics_app()
        client.get("/items/1")
        assert "campusmind_request_seconds_bucket" in client.get("/metrics").text

    def test_queue_length_comes_from_redis(self) -> None:
        redis_client = fakeredis.FakeRedis(decode_responses=True)
        redis_client.rpush("celery", "a", "b", "c")
        assert "campusmind_queue_length 3.0" in metrics_app(redis_client).get("/metrics").text

    def test_a_redis_outage_leaves_the_other_metrics_working(self) -> None:
        import redis

        class Broken:
            def llen(self, _: str) -> int:
                raise redis.ConnectionError("down")

        response = metrics_app(Broken()).get("/metrics")
        assert response.status_code == 200
        assert "campusmind_queue_length" not in response.text

    def test_the_real_app_exposes_metrics(self, client: TestClient) -> None:
        assert client.get("/metrics").status_code == 200


class TestReadiness:
    def test_all_healthy(self, client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(main_module, "_redis_ok", lambda: True)
        response = client.get("/readyz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "checks": {"postgres": True, "redis": True}}

    def test_a_failed_dependency_returns_503(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(main_module, "_redis_ok", lambda: False)
        response = client.get("/readyz")
        assert response.status_code == 503
        assert response.json()["status"] == "unavailable"


class TestSampleDocuments:
    def test_there_are_twenty(self) -> None:
        assert len(SAMPLES) == 20

    def test_every_sample_says_it_is_invented(self) -> None:
        assert all(BANNER in sample.text for sample in SAMPLES)

    def test_filenames_are_unique(self) -> None:
        assert len({sample.filename for sample in SAMPLES}) == 20

    def test_every_answer_is_in_its_own_document(self) -> None:
        for sample in SAMPLES:
            for _, answer in sample.questions:
                key = answer.split(",")[-1].strip().lower()
                assert key in sample.text.lower(), answer

    def test_every_document_type_is_represented(self) -> None:
        kinds = {sample.doc_type for sample in SAMPLES}
        assert kinds == {"syllabus", "circular", "notice", "regulation", "notes", "contact_list"}


class TestInfrastructureFiles:
    @pytest.mark.parametrize(
        "path",
        [
            "infra/compose/docker-compose.dev.yml",
            "infra/compose/docker-compose.prod.yml",
            ".github/workflows/ci.yml",
            "infra/prometheus/prometheus.yml",
            "infra/grafana/provisioning/datasources/prometheus.yml",
            "infra/grafana/provisioning/dashboards/campusmind.yml",
        ],
    )
    def test_yaml_parses(self, path: str) -> None:
        assert yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))

    def test_prod_uses_a_store_that_supports_several_processes(self) -> None:
        text = (ROOT / "infra/compose/docker-compose.prod.yml").read_text(encoding="utf-8")
        assert "VECTOR_BACKEND: qdrant" in text
        assert 'COOKIE_SECURE: "true"' in text

    def test_the_dashboard_covers_latency_errors_and_queue(self) -> None:
        dashboard = json.loads((ROOT / "infra/grafana/dashboards/campusmind.json").read_text())
        expressions = " ".join(
            target["expr"] for panel in dashboard["panels"] for target in panel["targets"]
        )
        for metric in (
            "campusmind_request_seconds_bucket",
            "campusmind_requests_total",
            "campusmind_queue_length",
        ):
            assert metric in expressions

    def test_metrics_are_not_reachable_from_outside(self) -> None:
        text = (ROOT / "infra/nginx/proxy.conf.template").read_text(encoding="utf-8")
        assert "location = /metrics" in text

    def test_images_do_not_run_as_root(self) -> None:
        backend = (ROOT / "infra/docker/backend.Dockerfile").read_text(encoding="utf-8")
        frontend = (ROOT / "infra/docker/frontend.Dockerfile").read_text(encoding="utf-8")
        assert "USER campusmind" in backend
        assert "nginx-unprivileged" in frontend

    def test_no_real_secrets_in_the_example_env(self) -> None:
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        assert "ANTHROPIC_API_KEY=\n" in text
        assert "POSTGRES_PASSWORD=\n" in text
