"""Tests for the admin analytics."""

import csv
import io
import uuid
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.analytics import service
from app.analytics.clustering import cluster_questions
from app.analytics.service import DateRange
from app.api.analytics import _csv_cell
from app.db import get_engine
from app.main import create_app
from app.models import Conversation, Feedback, Message, User
from app.vectorstore.embeddings import get_embedder
from tests.conftest import auth_header
from tests.helpers import HashingEmbedder
from tests.test_vectorstore import add_ready_document

JUNE_10 = datetime(2026, 6, 10, 6, 0, tzinfo=UTC)
PERIOD = DateRange(date(2026, 6, 1), date(2026, 6, 30))


class Workspace:
    """Adds conversations and messages at chosen times."""

    def __init__(self, db: Session, make_user: Callable[..., User]) -> None:
        self.db = db
        self.make_user = make_user

    def turn(
        self,
        user: User,
        question: str,
        *,
        at: datetime = JUNE_10,
        answered: bool | None = True,
        intent: str = "qa",
        latency_ms: int = 1000,
        tokens: tuple[int, int] = (100, 20),
        model: str = "llama3.2:1b",
        citations: list[dict[str, Any]] | None = None,
    ) -> Message:
        """Store a question and its answer."""
        conversation = Conversation(user_id=user.id)
        self.db.add(conversation)
        self.db.flush()
        self.db.add(
            Message(
                conversation_id=conversation.id,
                user_id=user.id,
                role="user",
                content=question,
                created_at=at,
            )
        )
        answer = Message(
            conversation_id=conversation.id,
            user_id=user.id,
            role="assistant",
            content="Answer",
            answered=answered,
            intent=intent,
            latency_ms=latency_ms,
            tokens_in=tokens[0],
            tokens_out=tokens[1],
            model_name=model,
            citations=citations or [],
            created_at=at + timedelta(seconds=2),
        )
        self.db.add(answer)
        self.db.commit()
        return answer

    def rate(self, message: Message, user: User, rating: int, at: datetime = JUNE_10) -> None:
        """Store a rating."""
        self.db.add(Feedback(message_id=message.id, user_id=user.id, rating=rating, created_at=at))
        self.db.commit()


@pytest.fixture
def work(db: Session, make_user: Callable[..., User]) -> Workspace:
    return Workspace(db, make_user)


class FixedEmbedder:
    """Returns chosen vectors, for exact control over similarity."""

    dimension = 2

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        rows = np.array([self.vectors[text] for text in texts], dtype=np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class TestClustering:
    def test_similar_questions_merge(self) -> None:
        embedder = FixedEmbedder(
            {"fee?": [1.0, 0.0], "cost?": [0.99, 0.05], "library?": [0.0, 1.0]}
        )
        clusters = cluster_questions(["fee?", "cost?", "library?"], embedder, 0.85)
        assert sorted(cluster.count for cluster in clusters) == [1, 2]

    def test_exactly_at_the_threshold_stays_apart(self) -> None:
        embedder = FixedEmbedder({"a": [1.0, 0.0], "b": [1.0, 0.0]})
        assert len(cluster_questions(["a", "b"], embedder, 1.5)) == 2

    def test_the_commonest_wording_represents_the_group(self) -> None:
        embedder = FixedEmbedder({"rare": [1.0, 0.0], "common": [1.0, 0.01]})
        clusters = cluster_questions(["rare", "common", "common"], embedder, 0.85)
        assert clusters[0].representative == "common"
        assert clusters[0].count == 3

    def test_case_and_spacing_count_as_the_same_question(self) -> None:
        clusters = cluster_questions(["Fee?", "fee?", " FEE? "], HashingEmbedder(), 0.85)
        assert [cluster.count for cluster in clusters] == [3]

    def test_largest_group_comes_first(self) -> None:
        embedder = FixedEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
        clusters = cluster_questions(["a", "b", "b"], embedder, 0.85)
        assert clusters[0].representative == "b"

    def test_examples_are_capped(self) -> None:
        questions = [f"same words {n}" for n in range(6)]
        embedder = FixedEmbedder({q: [1.0, 0.0] for q in questions})
        assert len(cluster_questions(questions, embedder, 0.85, max_examples=2)[0].examples) == 2

    def test_nothing_to_cluster(self) -> None:
        assert cluster_questions([], HashingEmbedder()) == []
        assert cluster_questions(["  "], HashingEmbedder()) == []


class TestReports:
    def test_daily_activity_counts_distinct_users(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        first, second = make_user(), make_user()
        work.turn(first, "a")
        work.turn(first, "b")
        work.turn(second, "c")
        work.turn(first, "d", at=JUNE_10 + timedelta(days=1))
        daily = service.daily_activity(db, PERIOD)
        assert daily == [
            {"day": date(2026, 6, 10), "active_users": 2, "questions": 3},
            {"day": date(2026, 6, 11), "active_users": 1, "questions": 1},
        ]

    def test_days_use_the_college_timezone(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        late_evening_utc = datetime(2026, 6, 10, 20, 0, tzinfo=UTC)
        work.turn(make_user(), "after midnight in India", at=late_evening_utc)
        assert service.daily_activity(db, PERIOD)[0]["day"] == date(2026, 6, 11)

    def test_the_range_includes_both_end_days(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        work.turn(user, "first day", at=datetime(2026, 6, 1, 0, 0, tzinfo=UTC))
        work.turn(user, "last day", at=datetime(2026, 6, 30, 12, 0, tzinfo=UTC))
        work.turn(user, "outside", at=datetime(2026, 7, 2, 0, 0, tzinfo=UTC))
        assert sum(row["questions"] for row in service.daily_activity(db, PERIOD)) == 2

    def test_overview_numbers(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        first = work.turn(user, "a", latency_ms=1000)
        second = work.turn(user, "b", latency_ms=3000)
        third = work.turn(user, "c", latency_ms=2000)
        work.rate(first, user, 1)
        work.rate(second, user, 1)
        work.rate(third, user, -1)
        report = service.overview(db, PERIOD)
        assert report["avg_latency_ms"] == 2000
        assert (report["thumbs_up"], report["thumbs_down"]) == (2, 1)
        assert report["feedback_ratio"] == pytest.approx(0.667, abs=0.001)

    def test_overview_with_no_data(self, db: Session) -> None:
        report = service.overview(db, PERIOD)
        assert report["avg_latency_ms"] is None
        assert report["feedback_ratio"] is None
        assert report["daily"] == []

    def test_intent_distribution(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        work.turn(user, "a", intent="qa")
        work.turn(user, "b", intent="qa")
        work.turn(user, "c", intent="reminder")
        assert service.intent_distribution(db, PERIOD) == [
            {"intent": "qa", "count": 2},
            {"intent": "reminder", "count": 1},
        ]

    def test_top_documents_counts_citations(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        popular, _ = add_ready_document(db, "x", title="Popular")
        rare, _ = add_ready_document(db, "y", title="Rare")
        user = make_user()
        work.turn(
            user,
            "a",
            citations=[{"document_id": str(popular)}, {"document_id": str(rare)}],
        )
        work.turn(user, "b", citations=[{"document_id": str(popular)}])
        report = service.top_documents(db, PERIOD)
        assert [(row["title"], row["citations"]) for row in report] == [("Popular", 2), ("Rare", 1)]

    def test_a_deleted_document_drops_out_of_the_ranking(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        work.turn(make_user(), "a", citations=[{"document_id": str(uuid.uuid4())}])
        assert service.top_documents(db, PERIOD) == []

    def test_cost_uses_each_models_price(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        work.turn(user, "a", model="claude-haiku-5-5", tokens=(1_000_000, 1_000_000))
        work.turn(user, "b", model="llama3.2:1b", tokens=(500, 500))
        rows = {row["model_name"]: row for row in service.token_cost(db, PERIOD)}
        assert rows["claude-haiku-5-5"]["cost_usd"] == pytest.approx(0.60)
        assert rows["llama3.2:1b"]["cost_usd"] == 0

    def test_unanswered_finds_the_question_that_was_asked(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        work.turn(user, "Answered fine", answered=True)
        work.turn(user, "Is there a swimming pool?", answered=False)
        rows, total = service.unanswered_questions(db, PERIOD)
        assert total == 1
        assert rows[0]["question"] == "Is there a swimming pool?"
        assert rows[0]["user_id"] == user.id

    def test_unanswered_picks_the_right_turn_in_a_long_conversation(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        answer = work.turn(user, "First question", answered=True)
        later = JUNE_10 + timedelta(minutes=5)
        for role, content, answered, seconds in (
            ("user", "Second question", None, 0),
            ("assistant", "Sorry", False, 2),
            ("user", "Third question", None, 10),
        ):
            db.add(
                Message(
                    conversation_id=answer.conversation_id,
                    user_id=user.id,
                    role=role,
                    content=content,
                    answered=answered,
                    created_at=later + timedelta(seconds=seconds),
                )
            )
        db.commit()
        rows, _ = service.unanswered_questions(db, PERIOD)
        assert rows[0]["question"] == "Second question"

    def test_top_questions_merge_and_rank(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        for _ in range(3):
            work.turn(user, "What is the exam fee?")
        work.turn(user, "Where is the library?")
        report = service.top_questions(db, PERIOD, HashingEmbedder())
        assert report[0]["question"] == "What is the exam fee?"
        assert report[0]["count"] == 3

    def test_knowledge_gaps_include_unanswered_and_disliked(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        work.turn(user, "Is there a swimming pool?", answered=False)
        disliked = work.turn(user, "What is the bus route?", answered=True)
        work.rate(disliked, user, -1)
        liked = work.turn(user, "What is the fee?", answered=True)
        work.rate(liked, user, 1)
        report = service.knowledge_gaps(db, PERIOD, HashingEmbedder())
        assert {row["topic"] for row in report} == {
            "Is there a swimming pool?",
            "What is the bus route?",
        }

    def test_document_status_counts(self, db: Session) -> None:
        add_ready_document(db, "x")
        assert service.document_status_counts(db) == [{"status": "ready", "count": 1}]


class TestIndexes:
    def test_the_daily_report_can_use_an_index(
        self, work: Workspace, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        for number in range(30):
            work.turn(user, f"q{number}")
        with get_engine().connect() as connection:
            connection.execute(text("SET enable_seqscan = off"))
            connection.execute(text("ANALYZE messages"))
            plan = connection.execute(
                text(
                    "EXPLAIN SELECT count(*) FROM messages "
                    "WHERE role = 'user' AND created_at >= '2026-06-01' "
                    "AND created_at < '2026-07-01'"
                )
            ).scalars()
            assert "ix_messages_role_created" in "\n".join(plan)


class TestApi:
    @pytest.fixture
    def api(self, db: Session) -> TestClient:
        app = create_app()
        app.dependency_overrides[get_embedder] = lambda: HashingEmbedder()
        return TestClient(app)

    def url(self, name: str, suffix: str = "") -> str:
        return f"/api/v1/admin/analytics/{name}?from=2026-06-01&to=2026-06-30{suffix}"

    @pytest.mark.parametrize(
        "name",
        [
            "overview",
            "top-questions",
            "unanswered",
            "intents",
            "top-documents",
            "cost",
            "knowledge-gaps",
            "document-status",
        ],
    )
    def test_every_report_returns_json_and_csv(
        self,
        api: TestClient,
        work: Workspace,
        make_user: Callable[..., User],
        name: str,
    ) -> None:
        work.turn(make_user(), "Is there a pool?", answered=False)
        headers = auth_header(make_user("admin"))
        assert api.get(self.url(name), headers=headers).status_code == 200
        response = api.get(self.url(name, "&format=csv"), headers=headers)
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "attachment" in response.headers["content-disposition"]

    def test_csv_has_a_header_and_rows(
        self, api: TestClient, work: Workspace, make_user: Callable[..., User]
    ) -> None:
        work.turn(make_user(), "x", intent="qa")
        response = api.get(
            self.url("intents", "&format=csv"), headers=auth_header(make_user("admin"))
        )
        assert list(csv.DictReader(io.StringIO(response.text))) == [{"intent": "qa", "count": "1"}]

    def test_an_empty_report_is_an_empty_csv(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = api.get(
            self.url("intents", "&format=csv"), headers=auth_header(make_user("admin"))
        )
        assert response.text == ""

    def test_a_reversed_range_is_refused(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = api.get(
            "/api/v1/admin/analytics/intents?from=2026-07-01&to=2026-06-01",
            headers=auth_header(make_user("admin")),
        )
        assert response.status_code == 422

    def test_the_default_range_is_thirty_days(
        self, api: TestClient, work: Workspace, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        work.turn(user, "recent", at=datetime.now(UTC) - timedelta(days=2))
        work.turn(user, "old", at=datetime.now(UTC) - timedelta(days=60))
        rows = api.get(
            "/api/v1/admin/analytics/intents", headers=auth_header(make_user("admin"))
        ).json()
        assert rows == [{"intent": "qa", "count": 1}]

    def test_unanswered_is_paginated(
        self, api: TestClient, work: Workspace, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        for number in range(3):
            work.turn(user, f"q{number}", answered=False)
        body = api.get(
            self.url("unanswered", "&limit=2"), headers=auth_header(make_user("admin"))
        ).json()
        assert (len(body["items"]), body["total"]) == (2, 3)


@pytest.mark.parametrize("value", ["=SUM(A1)", "+cmd", "-1+1", "@x"])
def test_csv_cells_cannot_become_spreadsheet_formulas(value: str) -> None:
    assert _csv_cell(value) == f"'{value}"


def test_csv_lists_are_joined() -> None:
    assert _csv_cell(["a", "b"]) == "a | b"
