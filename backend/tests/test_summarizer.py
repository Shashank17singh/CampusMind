"""Tests for the summarizer, including a 40-page document."""

import json
import re
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.agents.state import HandlerInput, UserContext
from app.agents.summarizer import (
    MAP_SYSTEM,
    MAX_CONCURRENT_CALLS,
    MAX_INLINE_CHUNKS,
    VERIFY_SYSTEM,
    Group,
    Item,
    SourceChunk,
    Summarizer,
    SummarizerAgent,
    SummaryContent,
    SummaryError,
    batch_chunks,
    detect_mode,
    formulas_in,
    has_unknown_formula,
    render_text,
    section_topics,
)
from app.agents.summary_jobs import chunks_from_file, run_summary, to_markdown, to_pdf
from app.config import get_settings
from app.db import get_session_factory
from app.ingestion.pipeline import PipelineDeps, run_ingestion
from app.ingestion.service import create_document
from app.llm import LLMError
from app.models import ChunkMeta, Document, Summary, User
from app.vectorstore.bm25 import BM25Index
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.hybrid import HybridRetriever
from tests.conftest import FIXTURES, auth_header
from tests.helpers import HashingEmbedder, RoutedLLM
from tests.test_vectorstore import DIMENSION, add_ready_document, embedded, record


def source(count: int = 4) -> list[SourceChunk]:
    """Build a small document."""
    return [
        SourceChunk(f"Page {n} explains topic {n}. The value of $E = mc^2$ is shown.", n)
        for n in range(1, count + 1)
    ]


def default_llm(reduce_items: list[dict] | None = None, drop: set[int] | None = None) -> RoutedLLM:
    """Build an LLM that plays each stage, chosen by the system prompt."""
    items = reduce_items or [{"text": "Topic one is explained.", "pages": [1]}]

    def respond(system: str, prompt: str) -> str:
        if system == MAP_SYSTEM:
            pages = sorted({int(page) for page in re.findall(r"\[page (\d+)\]", prompt)})
            return json.dumps({"points": [{"text": "A key point.", "pages": pages[:1]}]})
        if system == VERIFY_SYSTEM:
            numbers = re.findall(r"^(\d+)\. ", prompt.split("Statements:")[1], re.M)
            kept = [int(n) for n in numbers if int(n) not in (drop or set())]
            return json.dumps({"supported": kept})
        return json.dumps({"groups": [{"heading": "Summary", "items": items}]})

    return RoutedLLM(respond)


class TestHelpers:
    def test_batches_stay_under_the_limit_and_keep_pages(self) -> None:
        chunks = [SourceChunk("word " * 400, page) for page in range(1, 11)]
        batches = batch_chunks(chunks, max_tokens=1000)
        assert len(batches) > 1
        assert "[page 1]" in batches[0]
        assert "[page 10]" in batches[-1]

    def test_a_chunk_larger_than_a_batch_still_gets_its_own(self) -> None:
        assert len(batch_chunks([SourceChunk("word " * 2000, 1)], max_tokens=100)) == 1

    @pytest.mark.parametrize(
        ("text", "formula"),
        [
            ("Energy is $E = mc^2$ here", "E = mc^2"),
            ("Display $$a+b$$ math", "a+b"),
            (r"Inline \(x^2\) form", "x^2"),
            (r"Block \[y=mx\] form", "y=mx"),
        ],
    )
    def test_latex_delimiters_are_recognised(self, text: str, formula: str) -> None:
        assert formulas_in(text) == [formula]

    def test_a_formula_in_the_source_is_accepted_despite_spacing(self) -> None:
        assert not has_unknown_formula(Item(text="It is $E=mc^2$."), "ThevalueofE=mc^2isshown.")

    def test_an_invented_formula_is_caught(self) -> None:
        assert has_unknown_formula(Item(text="It is $F = ma$."), "E=mc^2")

    def test_formulas_in_flashcard_answers_are_checked(self) -> None:
        assert has_unknown_formula(Item(text="Q?", answer="$F=ma$"), "nothing")

    @pytest.mark.parametrize(
        ("message", "mode"),
        [
            ("summarize unit 2", "brief"),
            ("make flashcards from the OS notes", "flashcards"),
            ("give me exam notes", "exam_notes"),
            ("a detailed summary please", "detailed"),
        ],
    )
    def test_mode_comes_from_the_wording(self, message: str, mode: str) -> None:
        assert detect_mode(message) == mode

    def test_section_topics_use_the_deepest_heading_once(self) -> None:
        chunks = [
            SourceChunk("a", 1, "Unit 1 > Deadlocks"),
            SourceChunk("b", 2, "Unit 1 > Deadlocks"),
            SourceChunk("c", 3, "Unit 1 > Semaphores"),
            SourceChunk("d", 4),
        ]
        assert section_topics(chunks) == ["Deadlocks", "Semaphores"]


class TestSummarizer:
    def test_map_then_reduce_then_verify(self) -> None:
        llm = default_llm()
        result = Summarizer(llm).run(source(), "brief")
        assert result.content.groups[0].items[0].text == "Topic one is explained."
        assert result.verified is True
        assert result.llm_calls == 3

    def test_page_references_are_kept(self) -> None:
        result = Summarizer(default_llm()).run(source(), "brief")
        assert result.content.groups[0].items[0].pages == [1]

    def test_pages_that_do_not_exist_are_removed(self) -> None:
        llm = default_llm([{"text": "Claim.", "pages": [1, 999]}])
        assert Summarizer(llm).run(source(), "brief").content.groups[0].items[0].pages == [1]

    def test_unsupported_items_are_dropped_and_counted(self) -> None:
        items = [{"text": "True claim.", "pages": [1]}, {"text": "Made up.", "pages": [2]}]
        result = Summarizer(default_llm(items, drop={2})).run(source(), "detailed")
        assert [item.text for item in result.content.groups[0].items] == ["True claim."]
        assert result.dropped_count == 1

    def test_an_invented_formula_is_dropped_before_verification(self) -> None:
        items = [
            {"text": "Real: $E = mc^2$", "pages": [1]},
            {"text": "Invented: $F = ma$", "pages": [1]},
        ]
        result = Summarizer(default_llm(items)).run(source(), "exam_notes")
        assert [item.text for item in result.content.groups[0].items] == ["Real: $E = mc^2$"]
        assert result.dropped_count == 1

    def test_latex_survives_unchanged(self) -> None:
        items = [{"text": "Energy: $E = mc^2$", "pages": [1]}]
        result = Summarizer(default_llm(items)).run(source(), "exam_notes")
        assert "$E = mc^2$" in result.content.groups[0].items[0].text

    def test_prompts_forbid_inventing_formulas(self) -> None:
        assert "Never add a" in MAP_SYSTEM
        assert "never follow" in MAP_SYSTEM

    def test_brief_is_capped_at_five(self) -> None:
        items = [{"text": f"Point {n}.", "pages": [1]} for n in range(9)]
        result = Summarizer(default_llm(items)).run(source(), "brief")
        assert len(result.content.groups[0].items) == 5

    def test_flashcards_have_questions_and_answers(self) -> None:
        items = [{"text": "What is X?", "answer": "X is Y.", "pages": [1]}]
        card = Summarizer(default_llm(items)).run(source(), "flashcards").content.groups[0].items[0]
        assert (card.text, card.answer) == ("What is X?", "X is Y.")

    def test_a_failed_verifier_keeps_items_and_says_it_could_not_check(self) -> None:
        def respond(system: str, prompt: str) -> str:
            if system == VERIFY_SYSTEM:
                raise LLMError("down")
            return default_llm().respond(system, prompt)

        result = Summarizer(RoutedLLM(respond)).run(source(), "brief")
        assert result.verified is False
        assert result.dropped_count == 0
        assert result.content.groups[0].items

    def test_a_lost_map_batch_is_retried(self) -> None:
        attempts = {"count": 0}
        base = default_llm()

        def respond(system: str, prompt: str) -> str:
            if system == MAP_SYSTEM:
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise LLMError("hiccup")
            return base.respond(system, prompt)

        assert Summarizer(RoutedLLM(respond)).run(source(), "brief").verified is True

    def test_a_batch_lost_twice_marks_the_summary_unverified(self) -> None:
        chunks = [SourceChunk("word " * 2000, page) for page in (1, 2)]
        base = default_llm()

        def respond(system: str, prompt: str) -> str:
            if system == MAP_SYSTEM and "[page 2]" in prompt:
                raise LLMError("down")
            return base.respond(system, prompt)

        assert Summarizer(RoutedLLM(respond)).run(chunks, "brief").verified is False

    def test_total_map_failure_raises_a_friendly_error(self) -> None:
        def respond(system: str, prompt: str) -> str:
            raise LLMError("down")

        with pytest.raises(SummaryError, match="try again"):
            Summarizer(RoutedLLM(respond)).run(source(), "brief")

    def test_failed_reduce_raises_a_friendly_error(self) -> None:
        base = default_llm()

        def respond(system: str, prompt: str) -> str:
            if system not in (MAP_SYSTEM, VERIFY_SYSTEM):
                raise LLMError("down")
            return base.respond(system, prompt)

        with pytest.raises(SummaryError):
            Summarizer(RoutedLLM(respond)).run(source(), "brief")

    def test_an_empty_document_is_refused(self) -> None:
        with pytest.raises(SummaryError, match="no text"):
            Summarizer(default_llm()).run([], "brief")

    def test_points_too_long_for_one_prompt_are_condensed_in_rounds(self) -> None:
        long_point = "important detail " * 500

        def respond(system: str, prompt: str) -> str:
            if system == MAP_SYSTEM:
                text = "short point" if prompt.startswith("<source>\n- (pages") else long_point
                return json.dumps({"points": [{"text": text, "pages": [1]}]})
            return default_llm().respond(system, prompt)

        chunks = [SourceChunk("word " * 2400, page) for page in range(1, 7)]
        llm = RoutedLLM(respond)
        result = Summarizer(llm).run(chunks, "brief")
        assert result.content.groups[0].items
        assert result.llm_calls > 8


def make_forty_page_pdf(path: Path) -> None:
    """Write a 40-page PDF with distinct text on every page."""
    document = pymupdf.open()
    for number in range(1, 41):
        page = document.new_page()
        page.insert_text((72, 80), f"Chapter {number}", fontsize=18)
        body = " ".join(
            f"Section {number} sentence {n} describes concept {number * 100 + n} in detail."
            for n in range(1, 30)
        )
        page.insert_textbox(pymupdf.Rect(72, 110, 520, 760), body, fontsize=10)
    document.save(path)
    document.close()


class TestFortyPages:
    def test_forty_page_pdf(self, tmp_path: Path) -> None:
        path = tmp_path / "long.pdf"
        make_forty_page_pdf(path)
        chunks = chunks_from_file(path, "long.pdf", get_settings())
        assert {chunk.page for chunk in chunks} >= {1, 20, 40}

        llm = default_llm([{"text": f"Key idea {n}.", "pages": [n * 8]} for n in range(1, 6)])
        result = Summarizer(llm).run(chunks, "detailed")

        assert llm.peak <= MAX_CONCURRENT_CALLS
        assert llm.peak > 1, "Map calls should run in parallel"
        assert result.llm_calls == llm.calls
        assert result.tokens_in == llm.calls * 100
        assert result.tokens_out == llm.calls * 20
        assert result.duration_ms > 0
        assert result.content.groups[0].items
        assert result.verified is True


class TestExport:
    content = SummaryContent(
        groups=[
            Group(heading="Key Formulas", items=[Item(text="Energy: $E = mc^2$", pages=[3, 4])]),
            Group(
                heading="Flashcards",
                items=[Item(text="What is 2 < 3?", answer="True", pages=[1])],
            ),
        ]
    )

    def test_markdown(self) -> None:
        text = to_markdown("OS notes", self.content)
        assert text.startswith("# OS notes")
        assert "## Key Formulas" in text
        assert "- Energy: $E = mc^2$ (p. 3, 4)" in text
        assert "**Q:** What is 2 < 3?" in text

    def test_pdf_keeps_formulas_and_escapes_html(self) -> None:
        data = to_pdf("OS & Networks", self.content)
        document = pymupdf.open(stream=data, filetype="pdf")
        text = " ".join(page.get_text() for page in document)
        assert "$E = mc^2$" in text
        assert "OS & Networks" in text
        assert "2 < 3" in text

    def test_long_summaries_flow_onto_more_pages(self) -> None:
        items = [Item(text=f"Point number {n} has some text.") for n in range(200)]
        data = to_pdf("Long", SummaryContent(groups=[Group(heading="All", items=items)]))
        assert pymupdf.open(stream=data, filetype="pdf").page_count > 1

    def test_render_text_shows_no_empty_heading(self) -> None:
        text = render_text(SummaryContent(groups=[Group(items=[Item(text="Only point.")])]))
        assert text == "- Only point."


@pytest.fixture
def stored_document(db: Session) -> uuid.UUID:
    """A ready document with a chunk."""
    document_id, _ = add_ready_document(db, "The value of $E = mc^2$ is on this page.")
    return document_id


class TestJobs:
    def make(self, db: Session, user: User, document_id: uuid.UUID, mode: str = "brief") -> Summary:
        summary = Summary(user_id=user.id, document_id=document_id, source_name="Doc", mode=mode)
        db.add(summary)
        db.commit()
        return summary

    def test_a_job_stores_the_result(
        self, db: Session, stored_document: uuid.UUID, make_user: Callable[..., User]
    ) -> None:
        summary = self.make(db, make_user(), stored_document)
        assert run_summary(get_session_factory(), default_llm(), get_settings(), summary.id)
        db.expire_all()
        stored = db.get_one(Summary, summary.id)
        assert stored.status == "ready"
        assert stored.content["groups"][0]["items"]
        assert stored.content["verified"] is True
        assert stored.tokens_in > 0
        assert stored.duration_ms is not None

    def test_a_failed_job_records_a_friendly_message(
        self, db: Session, stored_document: uuid.UUID, make_user: Callable[..., User]
    ) -> None:
        def respond(system: str, prompt: str) -> str:
            raise LLMError("secret internal detail")

        summary = self.make(db, make_user(), stored_document)
        assert not run_summary(
            get_session_factory(), RoutedLLM(respond), get_settings(), summary.id
        )
        db.expire_all()
        stored = db.get_one(Summary, summary.id)
        assert stored.status == "failed"
        assert "try again" in stored.error_message
        assert "secret" not in stored.error_message

    def test_an_uploaded_file_is_summarized_then_deleted(
        self, db: Session, tmp_path: Path, make_user: Callable[..., User]
    ) -> None:
        path = tmp_path / "upload.docx"
        shutil.copy(FIXTURES / "tables.docx", path)
        summary = Summary(user_id=make_user().id, source_name="tables.docx", mode="brief")
        db.add(summary)
        db.commit()
        assert run_summary(get_session_factory(), default_llm(), get_settings(), summary.id, path)
        assert not path.exists()

    def test_an_upload_is_deleted_even_when_the_job_fails(
        self, db: Session, tmp_path: Path, make_user: Callable[..., User]
    ) -> None:
        path = tmp_path / "bad.pdf"
        shutil.copy(FIXTURES / "corrupted.pdf", path)
        summary = Summary(user_id=make_user().id, source_name="bad.pdf", mode="brief")
        db.add(summary)
        db.commit()
        assert not run_summary(
            get_session_factory(), default_llm(), get_settings(), summary.id, path
        )
        assert not path.exists()
        db.expire_all()
        assert "corrupted" in db.get_one(Summary, summary.id).error_message


class TestApi:
    @pytest.fixture(autouse=True)
    def _fake_llm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import app.agents.summary_jobs as jobs

        monkeypatch.setattr(jobs, "get_llm", lambda: default_llm())

    def test_summarize_a_stored_document(
        self,
        client: TestClient,
        stored_document: uuid.UUID,
        make_user: Callable[..., User],
    ) -> None:
        user = make_user()
        headers = auth_header(user)
        accepted = client.post(
            "/api/v1/summaries",
            headers=headers,
            json={"document_id": str(stored_document), "mode": "brief"},
        )
        assert accepted.status_code == 202
        summary = client.get(
            f"/api/v1/summaries/{accepted.json()['summary_id']}", headers=headers
        ).json()
        assert summary["status"] == "ready"
        assert summary["content"]["groups"]

    def test_an_unfinished_document_is_refused(
        self, client: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        document_id, _ = add_ready_document(db, "text")
        db.get_one(Document, document_id).status = "embedding"
        db.commit()
        response = client.post(
            "/api/v1/summaries",
            headers=auth_header(make_user()),
            json={"document_id": str(document_id), "mode": "brief"},
        )
        assert response.status_code == 409

    def test_an_unknown_document_is_404(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = client.post(
            "/api/v1/summaries",
            headers=auth_header(make_user()),
            json={"document_id": str(uuid.uuid4()), "mode": "brief"},
        )
        assert response.status_code == 404

    def test_an_unknown_mode_is_rejected(
        self, client: TestClient, stored_document: uuid.UUID, make_user: Callable[..., User]
    ) -> None:
        response = client.post(
            "/api/v1/summaries",
            headers=auth_header(make_user()),
            json={"document_id": str(stored_document), "mode": "poem"},
        )
        assert response.status_code == 422

    def test_uploads_never_enter_the_document_store(
        self, client: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        headers = auth_header(make_user())
        response = client.post(
            "/api/v1/summaries/upload",
            headers=headers,
            files={"file": ("notes.docx", (FIXTURES / "tables.docx").read_bytes())},
            data={"mode": "flashcards"},
        )
        assert response.status_code == 202
        summary = client.get(
            f"/api/v1/summaries/{response.json()['summary_id']}", headers=headers
        ).json()
        assert summary["status"] == "ready"
        assert summary["document_id"] is None
        assert summary["source_name"] == "notes.docx"
        assert db.scalar(select(func.count()).select_from(Document)) == 0
        assert db.scalar(select(func.count()).select_from(ChunkMeta)) == 0

    def test_a_bad_upload_is_rejected_at_once(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = client.post(
            "/api/v1/summaries/upload",
            headers=auth_header(make_user()),
            files={"file": ("run.exe", b"MZ")},
            data={"mode": "brief"},
        )
        assert response.status_code == 415

    @pytest.mark.parametrize(
        ("file_format", "media_type"), [("md", "text/markdown"), ("pdf", "application/pdf")]
    )
    def test_export(
        self,
        client: TestClient,
        stored_document: uuid.UUID,
        make_user: Callable[..., User],
        file_format: str,
        media_type: str,
    ) -> None:
        headers = auth_header(make_user())
        summary_id = client.post(
            "/api/v1/summaries",
            headers=headers,
            json={"document_id": str(stored_document), "mode": "brief"},
        ).json()["summary_id"]
        response = client.get(
            f"/api/v1/summaries/{summary_id}/export?format={file_format}", headers=headers
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith(media_type)

    def test_export_before_it_is_ready_is_refused(
        self, client: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user()
        summary = Summary(user_id=user.id, source_name="x", mode="brief")
        db.add(summary)
        db.commit()
        response = client.get(f"/api/v1/summaries/{summary.id}/export", headers=auth_header(user))
        assert response.status_code == 409

    def test_other_users_cannot_read_a_summary(
        self, client: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        summary = Summary(user_id=make_user().id, source_name="x", mode="brief")
        db.add(summary)
        db.commit()
        response = client.get(f"/api/v1/summaries/{summary.id}", headers=auth_header(make_user()))
        assert response.status_code == 404

    def test_list_shows_only_my_summaries(
        self, client: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        mine, other = make_user(), make_user()
        db.add_all(
            [
                Summary(user_id=mine.id, source_name="a", mode="brief"),
                Summary(user_id=other.id, source_name="b", mode="brief"),
            ]
        )
        db.commit()
        listed = client.get("/api/v1/summaries", headers=auth_header(mine)).json()
        assert [item["source_name"] for item in listed["items"]] == ["a"]


class TestAgent:
    def setup_agent(
        self, db: Session, tmp_path: Path, chunk_count: int = 2
    ) -> tuple[SummarizerAgent, list[tuple[uuid.UUID, str, uuid.UUID]]]:
        store = ChromaVectorStore(str(tmp_path / "chroma"))
        document_id, chunk_id = add_ready_document(db, "Deadlocks need four conditions.")
        for index in range(1, chunk_count):
            db.add(
                ChunkMeta(
                    document_id=document_id,
                    chunk_index=index,
                    text=f"More on deadlocks {index}.",
                    token_count=5,
                    page_number=index,
                    section_heading="Unit 3 > Deadlocks",
                )
            )
        db.commit()
        store.add_chunks(
            embedded(
                record("Deadlocks need four conditions.", id=chunk_id, document_id=document_id)
            )
        )
        retriever = HybridRetriever(
            store, HashingEmbedder(DIMENSION), BM25Index(get_session_factory())
        )
        queued: list[tuple[uuid.UUID, str, uuid.UUID]] = []
        agent = SummarizerAgent(
            default_llm(),
            retriever,
            get_session_factory(),
            lambda *args: queued.append(args),
        )
        return agent, queued

    def request(self, text: str) -> HandlerInput:
        return HandlerInput(text, (), UserContext(id=uuid.uuid4()), (), lambda _: None)

    def test_a_short_document_is_summarized_in_the_chat(self, db: Session, tmp_path: Path) -> None:
        agent, queued = self.setup_agent(db, tmp_path)
        result = agent.handle(self.request("summarize the deadlocks notes"))
        assert "Topic one" in result.text
        assert queued == []
        assert result.data["topics"] == ["Deadlocks"]

    def test_a_long_document_goes_to_the_background(self, db: Session, tmp_path: Path) -> None:
        agent, queued = self.setup_agent(db, tmp_path, chunk_count=MAX_INLINE_CHUNKS + 2)
        result = agent.handle(self.request("summarize the deadlocks notes"))
        assert "Summarizer page" in result.text
        assert len(queued) == 1
        assert result.data["topics"] == ["Deadlocks"]

    def test_no_match_is_said_plainly(self, db: Session, tmp_path: Path) -> None:
        store = ChromaVectorStore(str(tmp_path / "chroma"))
        retriever = HybridRetriever(
            store, HashingEmbedder(DIMENSION), BM25Index(get_session_factory())
        )
        agent = SummarizerAgent(default_llm(), retriever, get_session_factory(), lambda *_: None)
        assert "couldn't find" in agent.handle(self.request("summarize x")).text


def test_summaries_work_on_a_real_ingested_document(
    deps: PipelineDeps, db: Session, make_user: Callable[..., User]
) -> None:
    document = create_document(
        db,
        deps.storage,
        filename="text.pdf",
        data=(FIXTURES / "text.pdf").read_bytes(),
        uploader_id=None,
        doc_type="notes",
    ).document
    db.commit()
    run_ingestion(deps, document.id)
    summary = Summary(
        user_id=make_user().id, document_id=document.id, source_name="text", mode="brief"
    )
    db.add(summary)
    db.commit()
    assert run_summary(get_session_factory(), default_llm(), get_settings(), summary.id)


def test_the_database_rejects_an_unknown_summary_status(
    db: Session, make_user: Callable[..., User]
) -> None:
    db.add(Summary(user_id=make_user().id, source_name="x", mode="brief", status="vibing"))
    with pytest.raises(IntegrityError):
        db.commit()
    db.rollback()
