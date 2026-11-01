"""Tests for conversations, streaming turns, search, feedback and retention."""

import json
import threading
import uuid
from collections.abc import Callable, Iterator
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agents.graph import AgentRunner
from app.agents.runner import get_agent_runner
from app.agents.state import HandlerInput, HandlerResult, UserContext
from app.agents.tracing import TraceRecorder
from app.api.conversations import _stream_turn
from app.conversations import tasks
from app.conversations.service import (
    STOPPED_INTENT,
    clean_title,
    delete_expired,
    make_title,
)
from app.db import get_session_factory
from app.llm import ChatMessage, LLMError
from app.main import create_app
from app.models import AgentTrace, Conversation, Feedback, Message, User
from app.models.base import utcnow
from tests.conftest import auth_header
from tests.helpers import FakeLLM
from tests.test_rag import parse_sse

ROUTE_QA = '{"intents": ["qa"], "confidence": 0.9}'


def qa_handler(request: HandlerInput) -> HandlerResult:
    """Answer like the RAG agent, with one citation."""
    request.emit("The fee ")
    request.emit("is 800 [1].")
    rag = {
        "answer": "The fee is 800 [1].",
        "answered": True,
        "confidence": "high",
        "model": "test-model",
        "citations": [
            {"document_id": str(uuid.uuid4()), "title": "Fees", "page": 2, "snippet": "800"}
        ],
        "suggested_followups": ["When is it due?"],
    }
    return HandlerResult("The fee is 800 [1].", {"rag": rag}, tokens_in=30, tokens_out=9)


def build_runner(replies: list[str] | None = None) -> AgentRunner:
    """Build a real graph with a scripted router and QA agent."""
    llm = FakeLLM(replies if replies is not None else [ROUTE_QA] * 5)
    return AgentRunner(llm, {"qa": qa_handler}, TraceRecorder(get_session_factory()))


@pytest.fixture(autouse=True)
def _fake_title_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tasks, "get_llm", lambda: FakeLLM(["Examination fee details"]))


@pytest.fixture
def api(db: Session) -> Iterator[TestClient]:
    app = create_app()
    app.dependency_overrides[get_agent_runner] = lambda: build_runner()
    yield TestClient(app)


def start(api: TestClient, user: User, **body: object) -> dict:
    """Create a conversation."""
    response = api.post("/api/v1/conversations", headers=auth_header(user), json=body)
    assert response.status_code == 201
    return response.json()


def send(api: TestClient, user: User, conversation_id: str, content: str = "What is the fee?"):
    """Send a message and parse the event stream."""
    response = api.post(
        f"/api/v1/conversations/{conversation_id}/messages",
        headers=auth_header(user),
        json={"content": content},
    )
    assert response.status_code == 200
    return parse_sse(response.text)


def add_conversation(db: Session, user: User, **fields: object) -> Conversation:
    """Store a conversation directly."""
    conversation = Conversation(user_id=user.id, **fields)
    db.add(conversation)
    db.commit()
    return conversation


class TestTitles:
    def test_clean_title_caps_at_six_words(self) -> None:
        assert clean_title("One two three four five six seven eight") == (
            "One two three four five six"
        )

    def test_clean_title_removes_quotes_and_trailing_punctuation(self) -> None:
        assert clean_title('"Exam fee details."') == "Exam fee details"

    def test_make_title_uses_the_llm(self) -> None:
        assert make_title("What is the fee?", FakeLLM(["Fee details"])) == "Fee details"

    def test_make_title_survives_an_outage(self) -> None:
        title = make_title("What is the examination fee for this year?", FakeLLM([LLMError("x")]))
        assert title == "What is the examination fee for"

    def test_an_empty_llm_reply_falls_back_to_the_question(self) -> None:
        assert make_title("What is the fee?", FakeLLM(["   "])) == "What is the fee"


class TestConversations:
    def test_list_is_paginated_newest_first(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        now = utcnow()
        for number in range(5):
            add_conversation(
                db, user, title=f"c{number}", last_message_at=now + timedelta(minutes=number)
            )
        headers = auth_header(user)
        first = api.get("/api/v1/conversations?limit=2", headers=headers).json()
        second = api.get("/api/v1/conversations?limit=2&offset=2", headers=headers).json()
        last = api.get("/api/v1/conversations?limit=2&offset=4", headers=headers).json()
        assert first["total"] == 5
        assert [item["title"] for item in first["items"]] == ["c4", "c3"]
        assert [item["title"] for item in second["items"]] == ["c2", "c1"]
        assert [item["title"] for item in last["items"]] == ["c0"]

    def test_a_new_message_moves_a_conversation_to_the_top(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        older = start(api, user, title="older")
        start(api, user, title="newer")
        send(api, user, older["id"])
        listed = api.get("/api/v1/conversations", headers=auth_header(user)).json()
        assert listed["items"][0]["id"] == older["id"]

    def test_rename(self, api: TestClient, make_user: Callable[..., User]) -> None:
        user = make_user()
        conversation = start(api, user)
        response = api.patch(
            f"/api/v1/conversations/{conversation['id']}",
            headers=auth_header(user),
            json={"title": "Renamed"},
        )
        assert response.json()["title"] == "Renamed"

    def test_delete_removes_its_messages(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        conversation = start(api, user)
        send(api, user, conversation["id"])
        response = api.delete(
            f"/api/v1/conversations/{conversation['id']}", headers=auth_header(user)
        )
        assert response.status_code == 204
        assert db.scalar(select(func.count()).select_from(Message)) == 0


class TestStreamingTurn:
    def test_events_arrive_in_order(self, api: TestClient, make_user: Callable[..., User]) -> None:
        user = make_user()
        events = send(api, user, start(api, user)["id"])
        assert [name for name, _ in events] == ["meta", "token", "token", "citations", "done"]

    def test_tokens_rebuild_the_answer(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        events = send(api, user, start(api, user)["id"])
        text = "".join(data["text"] for name, data in events if name == "token")
        assert text == "The fee is 800 [1]."

    def test_citations_and_followups_are_sent(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        citations = dict(send(api, user, start(api, user)["id"]))["citations"]
        assert citations["citations"][0]["title"] == "Fees"
        assert citations["confidence"] == "high"
        assert citations["suggested_followups"] == ["When is it due?"]

    def test_every_field_is_stored_on_the_assistant_message(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        conversation = start(api, user)
        events = dict(send(api, user, conversation["id"]))
        stored = db.scalar(select(Message).where(Message.role == "assistant"))
        assert stored is not None
        assert str(stored.id) == events["meta"]["message_id"]
        assert str(stored.trace_id) == events["meta"]["trace_id"]
        assert stored.content == "The fee is 800 [1]."
        assert stored.citations[0]["title"] == "Fees"
        assert stored.intent == "qa"
        assert stored.answered is True
        assert stored.model_name == "test-model"
        assert (stored.tokens_in, stored.tokens_out) == (10 + 30, 5 + 9)
        assert stored.latency_ms is not None

    def test_the_trace_can_be_found_from_the_message(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        trace_id = dict(send(api, user, start(api, user)["id"]))["meta"]["trace_id"]
        nodes = db.scalars(
            select(AgentTrace.node)
            .where(AgentTrace.trace_id == uuid.UUID(trace_id))
            .order_by(AgentTrace.step)
        ).all()
        assert nodes == ["router", "qa", "finalize"]

    def test_the_user_message_is_stored_first(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        send(api, user, start(api, user)["id"], "Original question")
        rows = db.scalars(select(Message).order_by(Message.created_at)).all()
        assert [(row.role, row.content) for row in rows][0] == ("user", "Original question")

    def test_earlier_turns_reach_the_agent(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        seen: list[HandlerInput] = []

        def spy(request: HandlerInput) -> HandlerResult:
            seen.append(request)
            return HandlerResult("ok")

        runner = AgentRunner(
            FakeLLM([ROUTE_QA, ROUTE_QA]), {"qa": spy}, TraceRecorder(get_session_factory())
        )
        app = create_app()
        app.dependency_overrides[get_agent_runner] = lambda: runner
        client = TestClient(app)
        user = make_user()
        conversation = start(client, user)
        send(client, user, conversation["id"], "first question")
        send(client, user, conversation["id"], "second question")
        assert seen[1].message == "second question"
        assert [message.content for message in seen[1].history] == ["first question", "ok"]

    def test_only_recent_history_is_loaded(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        from app.conversations.service import recent_history

        user = make_user()
        conversation = add_conversation(db, user)
        base = utcnow()
        for number in range(10):
            db.add(
                Message(
                    conversation_id=conversation.id,
                    user_id=user.id,
                    role="user",
                    content=f"m{number}",
                    created_at=base + timedelta(seconds=number),
                )
            )
        db.commit()
        history = recent_history(db, conversation.id)
        assert [message.content for message in history] == [f"m{n}" for n in range(4, 10)]

    def test_a_failing_agent_still_ends_cleanly_and_is_stored(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        def explode(_: HandlerInput) -> HandlerResult:
            raise RuntimeError("boom")

        runner = AgentRunner(
            FakeLLM([ROUTE_QA]), {"qa": explode}, TraceRecorder(get_session_factory())
        )
        app = create_app()
        app.dependency_overrides[get_agent_runner] = lambda: runner
        client = TestClient(app)
        user = make_user()
        events = send(client, user, start(client, user)["id"])
        assert [name for name, _ in events] == ["meta", "citations", "done"]
        stored = db.scalar(select(Message).where(Message.role == "assistant"))
        assert stored is not None and "Sorry, something went wrong" in stored.content

    def test_stopping_early_keeps_the_partial_answer(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        release = threading.Event()

        def slow(request: HandlerInput) -> HandlerResult:
            request.emit("Partial answer")
            release.wait(5)
            return HandlerResult("Partial answer and the rest")

        runner = AgentRunner(
            FakeLLM([ROUTE_QA]), {"qa": slow}, TraceRecorder(get_session_factory())
        )
        user = make_user()
        conversation = add_conversation(db, user)
        message_id = uuid.uuid4()
        stream = _stream_turn(
            runner,
            conversation.id,
            UserContext(id=user.id),
            [ChatMessage("user", "x")],
            message_id,
        )
        assert next(stream).startswith("event: meta")
        assert "Partial answer" in next(stream)
        stream.close()
        release.set()

        stored = db.get(Message, message_id)
        assert stored is not None
        assert stored.content == "Partial answer"
        assert stored.intent == STOPPED_INTENT

    def test_an_empty_message_is_refused(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        response = api.post(
            f"/api/v1/conversations/{start(api, user)['id']}/messages",
            headers=auth_header(user),
            json={"content": ""},
        )
        assert response.status_code == 422


class TestAutoTitle:
    def titled(self, db: Session, conversation_id: str) -> str | None:
        db.expire_all()
        return db.get_one(Conversation, uuid.UUID(conversation_id)).title

    def test_the_first_question_titles_the_conversation(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        conversation = start(api, user)
        send(api, user, conversation["id"])
        assert self.titled(db, conversation["id"]) == "Examination fee details"

    def test_a_conversation_named_by_the_user_keeps_its_name(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        conversation = start(api, user, title="My name")
        send(api, user, conversation["id"])
        assert self.titled(db, conversation["id"]) == "My name"

    def test_later_messages_do_not_retitle(
        self,
        api: TestClient,
        db: Session,
        make_user: Callable[..., User],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        user = make_user()
        conversation = start(api, user)
        send(api, user, conversation["id"])
        monkeypatch.setattr(tasks, "get_llm", lambda: FakeLLM(["Second title"]))
        send(api, user, conversation["id"])
        assert self.titled(db, conversation["id"]) == "Examination fee details"


class TestSearch:
    def add(self, db: Session, user: User, text: str) -> None:
        conversation = add_conversation(db, user)
        db.add(Message(conversation_id=conversation.id, user_id=user.id, role="user", content=text))
        db.commit()

    def search(self, api: TestClient, user: User, query: str) -> dict:
        return api.get(
            "/api/v1/conversations/search", params={"q": query}, headers=auth_header(user)
        ).json()

    def test_finds_a_word_in_my_messages(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        self.add(db, user, "What is the attendance requirement?")
        self.add(db, user, "Hostel mess timings")
        result = self.search(api, user, "attendance")
        assert result["total"] == 1
        assert "<mark>attendance</mark>" in result["items"][0]["snippet"]

    def test_stemming_matches_other_forms(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        self.add(db, user, "The examinations start in December")
        assert self.search(api, user, "examination")["total"] == 1

    def test_never_returns_another_users_messages(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        self.add(db, make_user(), "Secret attendance question")
        assert self.search(api, make_user(), "attendance")["total"] == 0

    def test_search_syntax_cannot_cause_an_error(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        self.add(db, user, "attendance")
        for query in ["'", '"unclosed', "a & | b", "!!", "()"]:
            response = api.get(
                "/api/v1/conversations/search", params={"q": query}, headers=auth_header(user)
            )
            assert response.status_code == 200

    def test_an_empty_query_is_refused(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = api.get("/api/v1/conversations/search?q=", headers=auth_header(make_user()))
        assert response.status_code == 422


class TestFeedback:
    def message(self, api: TestClient, db: Session, user: User, role: str = "assistant") -> Message:
        send(api, user, start(api, user)["id"])
        stored = db.scalar(select(Message).where(Message.role == role))
        assert stored is not None
        return stored

    def rate(self, api: TestClient, user: User, message: Message, **body: object):
        return api.post(
            f"/api/v1/messages/{message.id}/feedback", headers=auth_header(user), json=body
        )

    def test_thumbs_up(self, api: TestClient, db: Session, make_user: Callable[..., User]) -> None:
        user = make_user()
        response = self.rate(api, user, self.message(api, db, user), rating=1)
        assert response.status_code == 200
        assert response.json()["rating"] == 1

    def test_thumbs_down_with_a_reason_and_comment(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        body = self.rate(
            api,
            user,
            self.message(api, db, user),
            rating=-1,
            reason="outdated",
            comment="That rule changed",
        ).json()
        assert (body["reason"], body["comment"]) == ("outdated", "That rule changed")

    def test_rating_again_replaces_the_rating(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        message = self.message(api, db, user)
        self.rate(api, user, message, rating=1)
        response = self.rate(api, user, message, rating=-1, reason="wrong")
        assert response.json()["rating"] == -1
        assert db.scalar(select(func.count()).select_from(Feedback)) == 1

    @pytest.mark.parametrize(
        "body",
        [
            {"rating": 5},
            {"rating": -1, "reason": "boring"},
            {"rating": 1, "reason": "wrong"},
        ],
    )
    def test_invalid_feedback_is_refused(
        self, api: TestClient, db: Session, make_user: Callable[..., User], body: dict
    ) -> None:
        user = make_user()
        assert self.rate(api, user, self.message(api, db, user), **body).status_code == 422

    def test_my_own_question_cannot_be_rated(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        assert (
            self.rate(api, user, self.message(api, db, user, "user"), rating=1).status_code == 422
        )

    def test_another_users_message_cannot_be_rated(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        message = self.message(api, db, make_user())
        assert self.rate(api, make_user(), message, rating=1).status_code == 404

    def test_admins_can_filter_feedback(
        self, api: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        self.rate(api, user, self.message(api, db, user), rating=-1, reason="wrong")
        headers = auth_header(make_user("admin"))
        assert api.get("/api/v1/admin/feedback?rating=-1", headers=headers).json()["total"] == 1
        assert api.get("/api/v1/admin/feedback?rating=1", headers=headers).json()["total"] == 0
        assert api.get("/api/v1/admin/feedback?reason=wrong", headers=headers).json()["total"] == 1


class TestPrivacy:
    @pytest.mark.parametrize(
        ("method", "suffix", "body"),
        [
            ("GET", "/messages", None),
            ("PATCH", "", {"title": "Mine now"}),
            ("DELETE", "", None),
            ("POST", "/messages", {"content": "hi"}),
        ],
    )
    def test_users_cannot_touch_each_others_conversations(
        self,
        api: TestClient,
        make_user: Callable[..., User],
        method: str,
        suffix: str,
        body: dict | None,
    ) -> None:
        conversation = start(api, make_user())
        response = api.request(
            method,
            f"/api/v1/conversations/{conversation['id']}{suffix}",
            headers=auth_header(make_user()),
            json=body,
        )
        assert response.status_code == 404

    def test_the_list_shows_only_my_conversations(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        start(api, make_user(), title="theirs")
        mine = make_user()
        start(api, mine, title="mine")
        listed = api.get("/api/v1/conversations", headers=auth_header(mine)).json()
        assert [item["title"] for item in listed["items"]] == ["mine"]

    def test_export_contains_only_my_history(
        self, api: TestClient, make_user: Callable[..., User]
    ) -> None:
        other = make_user()
        send(api, other, start(api, other)["id"], "Their private question")
        mine = make_user()
        send(api, mine, start(api, mine)["id"], "My question")
        response = api.get("/api/v1/conversations/export", headers=auth_header(mine))
        assert "attachment" in response.headers["content-disposition"]
        data = json.loads(response.text)
        assert len(data["conversations"]) == 1
        messages = data["conversations"][0]["messages"]
        assert messages[0]["content"] == "My question"
        assert messages[1]["citations"][0]["title"] == "Fees"
        assert "Their private question" not in response.text


class TestRetention:
    def test_old_conversations_are_deleted_with_their_messages(
        self, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        old = add_conversation(db, user, last_message_at=utcnow() - timedelta(days=100))
        kept = add_conversation(db, user, last_message_at=utcnow() - timedelta(days=10))
        db.add(Message(conversation_id=old.id, user_id=user.id, role="user", content="x"))
        db.commit()

        assert delete_expired(db, retention_days=90) == 1
        db.commit()
        db.expire_all()
        assert db.get(Conversation, kept.id) is not None
        assert db.get(Conversation, old.id) is None
        assert db.scalar(select(func.count()).select_from(Message)) == 0

    def test_zero_days_keeps_everything(self, db: Session, make_user: Callable[..., User]) -> None:
        add_conversation(db, make_user(), last_message_at=utcnow() - timedelta(days=9999))
        assert delete_expired(db, retention_days=0) == 0

    def test_the_task_uses_the_configured_period(
        self, db: Session, make_user: Callable[..., User], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.config import get_settings

        add_conversation(db, make_user(), last_message_at=utcnow() - timedelta(days=500))
        monkeypatch.setattr(get_settings(), "conversation_retention_days", 365)
        assert tasks.delete_expired_conversations() == 1
