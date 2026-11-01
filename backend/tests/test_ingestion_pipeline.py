"""End-to-end ingestion tests against real files and a real database."""

import uuid
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.ingestion import tasks
from app.ingestion.errors import CorruptedFileError, ExtractionError, OcrUnavailableError
from app.ingestion.extract import extract
from app.ingestion.ocr import TesseractOcr
from app.ingestion.pipeline import STALE_CLAIM, PipelineDeps, run_ingestion
from app.ingestion.service import create_document
from app.ingestion.storage import LocalStorage
from app.ingestion.validation import FileKind
from app.models import AuditLog, ChunkMeta, Document, User
from app.models.base import utcnow
from tests.conftest import FIXTURES, auth_header
from tests.helpers import FakeIndexer, FakeOcr


def upload(
    deps: PipelineDeps, db: Session, filename: str, data: bytes, **fields: object
) -> uuid.UUID:
    """Create a document the way the API does."""
    result = create_document(
        db, deps.storage, filename=filename, data=data, uploader_id=None, **fields
    )
    db.commit()
    return result.document.id


def upload_fixture(deps: PipelineDeps, db: Session, name: str, **fields: object) -> uuid.UUID:
    """Create a document from a file in tests/fixtures."""
    return upload(deps, db, name, (FIXTURES / name).read_bytes(), **fields)


def chunks_of(db: Session, document_id: uuid.UUID) -> list[ChunkMeta]:
    """Return a document's chunks in order."""
    return list(
        db.scalars(
            select(ChunkMeta)
            .where(ChunkMeta.document_id == document_id)
            .order_by(ChunkMeta.chunk_index)
        )
    )


class TestExtraction:
    def test_text_pdf_keeps_pages_and_finds_the_table(self) -> None:
        result = extract(FIXTURES / "text.pdf", FileKind.PDF, FakeOcr())
        assert result.page_count == 4
        assert {block.page for block in result.blocks} == {1, 2, 3, 4}
        tables = [block for block in result.blocks if block.is_table]
        assert len(tables) == 1
        assert tables[0].page == 2
        assert "| Course | Credits |" in tables[0].text
        assert "| DBMS | 4 |" in tables[0].text

    def test_table_text_is_not_duplicated_outside_the_table(self) -> None:
        result = extract(FIXTURES / "text.pdf", FileKind.PDF, FakeOcr())
        plain = " ".join(block.text for block in result.blocks if not block.is_table)
        assert "Operating Systems" not in plain

    def test_large_text_is_recognised_as_a_heading(self) -> None:
        result = extract(FIXTURES / "text.pdf", FileKind.PDF, FakeOcr())
        assert any(
            block.heading_level == 1 and block.text.startswith("Unit 1") for block in result.blocks
        )

    def test_text_pages_do_not_use_ocr(self) -> None:
        ocr = FakeOcr()
        extract(FIXTURES / "text.pdf", FileKind.PDF, ocr)
        assert ocr.calls == 0

    def test_scanned_pdf_goes_through_ocr_with_its_page_number(self) -> None:
        ocr = FakeOcr("The library will remain closed on Friday.")
        result = extract(FIXTURES / "scanned.pdf", FileKind.PDF, ocr, dpi=72)
        assert ocr.calls == 1
        assert result.blocks[0].text == "The library will remain closed on Friday."
        assert result.blocks[0].page == 1

    def test_docx_keeps_tables_in_document_order(self) -> None:
        result = extract(FIXTURES / "tables.docx", FileKind.DOCX, FakeOcr())
        kinds = [(block.is_table, block.heading_level) for block in result.blocks]
        assert kinds == [(False, 1), (False, None), (False, 2), (True, None), (False, None)]
        assert "| Grade | Marks |" in result.blocks[3].text

    def test_pptx_keeps_slide_numbers_titles_tables_and_notes(self) -> None:
        result = extract(FIXTURES / "slides.pptx", FileKind.PPTX, FakeOcr())
        assert result.page_count == 2
        assert result.blocks[0].heading_level == 1
        assert any(block.is_table and block.page == 2 for block in result.blocks)
        assert any("starvation" in block.text and block.page == 1 for block in result.blocks)

    def test_image_is_read_with_ocr(self) -> None:
        result = extract(
            FIXTURES / "notice.png", FileKind.IMAGE, FakeOcr("Fee payment closes soon.")
        )
        assert result.blocks[0].text == "Fee payment closes soon."

    def test_txt_falls_back_to_windows_encoding(self, tmp_path: Path) -> None:
        path = tmp_path / "notice.txt"
        path.write_bytes("Café timings".encode("cp1252"))
        assert extract(path, FileKind.TXT, FakeOcr()).blocks[0].text == "Café timings"

    @pytest.mark.parametrize(
        ("name", "kind"), [("corrupted.pdf", FileKind.PDF), ("corrupted.docx", FileKind.DOCX)]
    )
    def test_corrupted_files_raise_a_clear_error(self, name: str, kind: FileKind) -> None:
        with pytest.raises(CorruptedFileError, match="corrupted"):
            extract(FIXTURES / name, kind, FakeOcr())

    def test_encrypted_pdf_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "locked.pdf"
        document = pymupdf.open()
        document.new_page().insert_text((72, 72), "secret")
        document.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="pw", owner_pw="own")
        document.close()
        with pytest.raises(CorruptedFileError, match="password"):
            extract(path, FileKind.PDF, FakeOcr())

    def test_scanned_pdf_without_tesseract_explains_why_it_failed(self) -> None:
        class MissingTesseract:
            def image_to_text(self, image: object) -> str:
                raise OcrUnavailableError("no tesseract")

        with pytest.raises(ExtractionError, match="needs OCR"):
            extract(FIXTURES / "scanned.pdf", FileKind.PDF, MissingTesseract(), dpi=72)

    @pytest.mark.ocr
    def test_real_tesseract_reads_the_scanned_pdf(self) -> None:
        import shutil

        if shutil.which("tesseract") is None:
            pytest.skip("Tesseract is not installed")
        result = extract(FIXTURES / "scanned.pdf", FileKind.PDF, TesseractOcr())
        assert "library" in " ".join(block.text for block in result.blocks).lower()


class TestPipeline:
    def test_text_pdf_reaches_ready_with_metadata_on_every_chunk(
        self, deps: PipelineDeps, db: Session
    ) -> None:
        document_id = upload_fixture(
            deps, db, "text.pdf", doc_type="notes", department="CSE", semester=3
        )
        assert run_ingestion(deps, document_id) is True

        document = db.get_one(Document, document_id)
        assert (document.status, document.progress, document.page_count) == ("ready", 100, 4)

        records = deps.indexer.records[document_id]
        assert records
        for record in records:
            assert record.document_id == document_id
            assert record.title == "text"
            assert record.page_number in {1, 2, 3, 4}
            assert (record.department, record.semester, record.doc_type) == ("CSE", 3, "notes")
            assert record.uploaded_at == document.uploaded_at

    def test_headers_and_page_numbers_do_not_reach_chunks(
        self, deps: PipelineDeps, db: Session
    ) -> None:
        document_id = upload_fixture(deps, db, "text.pdf", doc_type="notes")
        run_ingestion(deps, document_id)
        text = " ".join(chunk.text for chunk in chunks_of(db, document_id))
        assert "Government College" not in text
        assert "Page 2 of 4" not in text
        assert "Normal forms reduce redundancy" in text

    def test_table_arrives_as_one_chunk_on_the_right_page(
        self, deps: PipelineDeps, db: Session
    ) -> None:
        document_id = upload_fixture(deps, db, "text.pdf", doc_type="notes")
        run_ingestion(deps, document_id)
        tables = [chunk for chunk in chunks_of(db, document_id) if chunk.is_table]
        assert len(tables) == 1
        assert tables[0].page_number == 2
        assert "| DBMS | 4 |" in tables[0].text

    def test_scanned_pdf_is_ingested_through_ocr(self, deps: PipelineDeps, db: Session) -> None:
        deps.settings = deps.settings.model_copy(update={"ocr_dpi": 72})
        document_id = upload_fixture(deps, db, "scanned.pdf", doc_type="notice")
        assert run_ingestion(deps, document_id) is True
        assert "library" in chunks_of(db, document_id)[0].text

    def test_docx_with_tables_is_ingested(self, deps: PipelineDeps, db: Session) -> None:
        document_id = upload_fixture(deps, db, "tables.docx", doc_type="regulation")
        assert run_ingestion(deps, document_id) is True
        chunks = chunks_of(db, document_id)
        assert any(chunk.is_table and "| A | 90 |" in chunk.text for chunk in chunks)
        assert chunks[-1].section_heading == "Examination Rules > Grading"

    def test_every_status_is_visited_in_order(
        self, deps: PipelineDeps, db: Session, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import app.ingestion.pipeline as pipeline

        seen: list[tuple[str, int]] = []
        real = pipeline._set_status

        def spy(
            deps_: PipelineDeps,
            id_: uuid.UUID,
            status: str,
            progress: int,
            error: str | None = None,
        ) -> None:
            seen.append((status, progress))
            real(deps_, id_, status, progress, error)

        monkeypatch.setattr(pipeline, "_set_status", spy)
        run_ingestion(deps, upload_fixture(deps, db, "text.pdf", doc_type="notes"))

        statuses = [status for status, _ in seen]
        assert statuses[0] == "chunking"
        assert statuses[-1] == "ready"
        assert "embedding" in statuses
        progress = [value for _, value in seen]
        assert progress == sorted(progress)

    def test_corrupted_file_fails_with_a_readable_message(
        self, deps: PipelineDeps, db: Session
    ) -> None:
        document_id = upload_fixture(deps, db, "corrupted.pdf")
        assert run_ingestion(deps, document_id) is False
        document = db.get_one(Document, document_id)
        assert document.status == "failed"
        assert document.progress == 0
        assert "corrupted" in document.error_message

    def test_indexer_crash_marks_failed_and_leaves_nothing_behind(
        self, deps: PipelineDeps, db: Session
    ) -> None:
        deps.indexer = FakeIndexer(fail_with=RuntimeError("vector store exploded"))
        document_id = upload_fixture(deps, db, "text.pdf", doc_type="notes")
        assert run_ingestion(deps, document_id) is False
        document = db.get_one(Document, document_id)
        assert document.status == "failed"
        assert "exploded" not in document.error_message
        assert chunks_of(db, document_id) == []

    def test_database_outage_is_raised_so_celery_can_retry(
        self, deps: PipelineDeps, db: Session
    ) -> None:
        deps.indexer = FakeIndexer(fail_with=OperationalError("x", {}, Exception("db down")))
        document_id = upload_fixture(deps, db, "text.pdf", doc_type="notes")
        with pytest.raises(OperationalError):
            run_ingestion(deps, document_id)

    def test_a_running_document_cannot_be_claimed_twice(
        self, deps: PipelineDeps, db: Session
    ) -> None:
        document_id = upload_fixture(deps, db, "text.pdf", doc_type="notes")
        document = db.get_one(Document, document_id)
        document.status = "extracting"
        db.commit()
        assert run_ingestion(deps, document_id) is False
        assert document_id not in deps.indexer.records

    def test_an_abandoned_run_can_be_claimed_again(self, deps: PipelineDeps, db: Session) -> None:
        document_id = upload_fixture(deps, db, "text.pdf", doc_type="notes")
        document = db.get_one(Document, document_id)
        document.status = "embedding"
        document.updated_at = utcnow() - STALE_CLAIM - timedelta(minutes=1)
        db.commit()
        assert run_ingestion(deps, document_id) is True

    def test_a_ready_document_is_not_processed_again_by_a_redelivered_task(
        self, deps: PipelineDeps, db: Session
    ) -> None:
        document_id = upload_fixture(deps, db, "text.pdf", doc_type="notes")
        run_ingestion(deps, document_id)
        document = db.get_one(Document, document_id)
        document.updated_at = utcnow() - STALE_CLAIM - timedelta(minutes=1)
        db.commit()
        assert run_ingestion(deps, document_id) is False

    def test_reingesting_replaces_chunks_and_vectors(self, deps: PipelineDeps, db: Session) -> None:
        document_id = upload_fixture(deps, db, "text.pdf", doc_type="notes")
        run_ingestion(deps, document_id)
        first_count = len(chunks_of(db, document_id))
        document = db.get_one(Document, document_id)
        document.status = "uploaded"
        db.commit()
        run_ingestion(deps, document_id)
        db.expire_all()
        assert len(chunks_of(db, document_id)) == first_count
        assert len(deps.indexer.records[document_id]) == first_count

    def test_on_ready_hook_runs_after_success(self, deps: PipelineDeps, db: Session) -> None:
        ready: list[uuid.UUID] = []
        deps.on_ready = ready.append
        document_id = upload_fixture(deps, db, "text.pdf", doc_type="notes")
        run_ingestion(deps, document_id)
        assert ready == [document_id]


CIRCULAR = " ".join(
    f"Clause {n} states that all students must follow the updated examination procedure."
    for n in range(1, 60)
)


class TestDeduplication:
    def test_duplicate_upload_is_skipped(self, deps: PipelineDeps, db: Session) -> None:
        data = (FIXTURES / "text.pdf").read_bytes()
        first = create_document(db, deps.storage, filename="a.pdf", data=data, uploader_id=None)
        db.commit()
        second = create_document(
            db, deps.storage, filename="renamed.pdf", data=data, uploader_id=None
        )
        assert second.duplicate is True
        assert second.document.id == first.document.id
        assert db.scalar(select(Document.id).where(Document.filename == "renamed.pdf")) is None

    def test_reuploading_a_failed_file_retries_it(self, deps: PipelineDeps, db: Session) -> None:
        data = b"Failed once."
        document_id = upload(deps, db, "a.txt", data)
        document = db.get_one(Document, document_id)
        document.status, document.error_message = "failed", "boom"
        db.commit()
        result = create_document(db, deps.storage, filename="a.txt", data=data, uploader_id=None)
        assert result.duplicate is False
        assert (result.document.status, result.document.error_message) == ("uploaded", None)

    def test_new_circular_version_supersedes_the_old_one(
        self, deps: PipelineDeps, db: Session
    ) -> None:
        old = upload(deps, db, "circular_v1.txt", CIRCULAR.encode(), doc_type="circular")
        run_ingestion(deps, old)
        new_text = CIRCULAR.replace("Clause 30 ", "Clause 30 (amended) ")
        new = upload(deps, db, "circular_v2.txt", new_text.encode(), doc_type="circular")
        run_ingestion(deps, new)

        db.expire_all()
        assert db.get_one(Document, old).superseded_by_id == new
        assert db.get_one(Document, new).superseded_by_id is None
        assert deps.indexer.superseded == {old}

    def test_unrelated_circular_supersedes_nothing(self, deps: PipelineDeps, db: Session) -> None:
        old = upload(deps, db, "c1.txt", CIRCULAR.encode(), doc_type="circular")
        run_ingestion(deps, old)
        other = " ".join(f"Hostel rule {n} covers visiting hours on weekends." for n in range(60))
        run_ingestion(deps, upload(deps, db, "c2.txt", other.encode(), doc_type="circular"))
        db.expire_all()
        assert db.get_one(Document, old).superseded_by_id is None

    def test_notes_are_never_superseded(self, deps: PipelineDeps, db: Session) -> None:
        old = upload(deps, db, "n1.txt", CIRCULAR.encode(), doc_type="notes")
        run_ingestion(deps, old)
        run_ingestion(
            deps, upload(deps, db, "n2.txt", (CIRCULAR + " Extra.").encode(), doc_type="notes")
        )
        db.expire_all()
        assert db.get_one(Document, old).superseded_by_id is None

    def test_circulars_from_other_departments_are_left_alone(
        self, deps: PipelineDeps, db: Session
    ) -> None:
        old = upload(deps, db, "c1.txt", CIRCULAR.encode(), doc_type="circular", department="CSE")
        run_ingestion(deps, old)
        new = upload(
            deps,
            db,
            "c2.txt",
            (CIRCULAR + " Extra.").encode(),
            doc_type="circular",
            department="ECE",
        )
        run_ingestion(deps, new)
        db.expire_all()
        assert db.get_one(Document, old).superseded_by_id is None

    def test_a_failed_new_version_does_not_hide_the_old_one(
        self, deps: PipelineDeps, db: Session
    ) -> None:
        old = upload(deps, db, "c1.txt", CIRCULAR.encode(), doc_type="circular")
        run_ingestion(deps, old)
        deps.indexer = FakeIndexer(fail_with=RuntimeError("down"))
        run_ingestion(
            deps, upload(deps, db, "c2.txt", (CIRCULAR + " Extra.").encode(), doc_type="circular")
        )
        db.expire_all()
        assert db.get_one(Document, old).superseded_by_id is None


class TestDocumentApi:
    @pytest.fixture(autouse=True)
    def _use_fake_deps(self, deps: PipelineDeps, monkeypatch: pytest.MonkeyPatch) -> None:
        deps.storage = LocalStorage(get_settings().upload_dir)
        monkeypatch.setattr(tasks, "build_deps", lambda: deps)

    def _post(self, client: TestClient, user: User, name: str = "tables.docx", **form: str):
        content = (FIXTURES / name).read_bytes()
        return client.post(
            "/api/v1/documents",
            headers=auth_header(user),
            files={"file": (name, content)},
            data=form,
        )

    def test_faculty_upload_runs_the_whole_pipeline(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = self._post(client, make_user("faculty"), doc_type="regulation")
        assert response.status_code == 202
        body = response.json()
        assert (body["status"], body["progress"], body["duplicate_of"]) == ("ready", 100, None)

    def test_upload_writes_an_audit_entry(
        self, client: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        user = make_user("faculty")
        self._post(client, user, doc_type="regulation")
        entry = db.scalar(select(AuditLog).where(AuditLog.action == "document.upload"))
        assert entry is not None and entry.actor_id == user.id

    def test_a_client_address_that_is_not_an_ip_does_not_break_the_upload(
        self, client: TestClient, db: Session, make_user: Callable[..., User]
    ) -> None:
        response = self._post(client, make_user("faculty"), doc_type="regulation")
        assert response.status_code == 202
        entry = db.scalar(select(AuditLog).where(AuditLog.action == "document.upload"))
        assert entry is not None and entry.ip_address is None

    def test_students_cannot_upload(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = self._post(client, make_user("student"))
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "forbidden"

    def test_upload_needs_a_token(self, client: TestClient) -> None:
        response = client.post("/api/v1/documents", files={"file": ("a.txt", b"x")})
        assert response.status_code == 401

    def test_duplicate_upload_points_at_the_original(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = make_user("faculty")
        first = self._post(client, user, doc_type="regulation").json()
        second = self._post(client, user).json()
        assert second["duplicate_of"] == first["id"]

    def test_unsupported_file_gets_415_and_a_clear_message(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = client.post(
            "/api/v1/documents",
            headers=auth_header(make_user("faculty")),
            files={"file": ("run.exe", b"MZ")},
        )
        assert response.status_code == 415
        error = response.json()["error"]
        assert error["code"] == "unsupported_file_type"
        assert "PDF" in error["message"]
        assert error["trace_id"]

    def test_oversized_upload_gets_413(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = client.post(
            "/api/v1/documents",
            headers=auth_header(make_user("faculty")),
            files={"file": ("big.txt", b"x" * (25 * 1024 * 1024 + 1))},
        )
        assert response.status_code == 413

    def test_invalid_doc_type_is_rejected(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = self._post(client, make_user("faculty"), doc_type="poem")
        assert response.status_code == 422

    def test_corrupted_upload_reports_failure_through_status(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        user = make_user("faculty")
        document_id = self._post(client, user, "corrupted.pdf").json()["id"]
        status = client.get(
            f"/api/v1/documents/{document_id}/status", headers=auth_header(user)
        ).json()
        assert status["status"] == "failed"
        assert "corrupted" in status["error_message"]

    def test_status_of_unknown_document_is_404(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        response = client.get(
            f"/api/v1/documents/{uuid.uuid4()}/status", headers=auth_header(make_user())
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_list_filters_and_paginates(
        self, client: TestClient, make_user: Callable[..., User]
    ) -> None:
        faculty = make_user("faculty")
        self._post(client, faculty, "tables.docx", doc_type="regulation")
        self._post(client, faculty, "text.pdf", doc_type="notes")
        headers = auth_header(make_user())
        everything = client.get("/api/v1/documents", headers=headers).json()
        assert everything["total"] == 2
        filtered = client.get("/api/v1/documents?doc_type=notes", headers=headers).json()
        assert [item["doc_type"] for item in filtered["items"]] == ["notes"]
        page = client.get("/api/v1/documents?limit=1&offset=1", headers=headers).json()
        assert (len(page["items"]), page["total"]) == (1, 2)


def test_health_endpoints(client: TestClient) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").json()["checks"]["postgres"] is True
