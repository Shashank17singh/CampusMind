"""Tests for the vector stores, embeddings, BM25, RRF and the hybrid retriever."""

import importlib.util
import sys
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy.orm import Session

from app.db import get_session_factory
from app.ingestion.pipeline import PipelineDeps, run_ingestion
from app.ingestion.service import create_document
from app.ingestion.types import ChunkRecord
from app.models import ChunkMeta, Document
from app.vectorstore.base import EmbeddedChunk, ScoredChunk, SearchFilters, VectorStore
from app.vectorstore.bm25 import BM25Index, tokenize
from app.vectorstore.chroma_store import ChromaVectorStore
from app.vectorstore.embeddings import (
    QUERY_PREFIX,
    SentenceTransformerEmbedder,
    split_into_windows,
)
from app.vectorstore.hybrid import HybridRetriever, RetrievalResult, SearchMode, rrf_merge
from app.vectorstore.indexer import VectorIndexer, embedding_text
from app.vectorstore.qdrant_store import QdrantVectorStore
from tests.helpers import HashingEmbedder

DIMENSION = 64


def record(text: str = "text", **overrides: object) -> ChunkRecord:
    """Build a chunk record."""
    fields: dict[str, object] = {
        "id": uuid.uuid4(),
        "document_id": uuid.uuid4(),
        "title": "Doc",
        "chunk_index": 0,
        "text": text,
        "token_count": 5,
        "page_number": 3,
        "section_heading": "Unit 1",
        "is_table": False,
        "department": "CSE",
        "semester": 3,
        "doc_type": "notes",
        "uploaded_at": datetime(2026, 7, 1, tzinfo=UTC),
        "uploader_id": None,
        **overrides,
    }
    return ChunkRecord(**fields)  # type: ignore[arg-type]


def embedded(*records: ChunkRecord) -> list[EmbeddedChunk]:
    """Attach hashed embeddings to records."""
    embedder = HashingEmbedder(DIMENSION)
    vectors = embedder.embed_documents([r.text for r in records])
    return [EmbeddedChunk(r, v) for r, v in zip(records, vectors, strict=True)]


@pytest.fixture(params=["chroma", "qdrant"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> VectorStore:
    """Each test runs against both real backends."""
    if request.param == "chroma":
        return ChromaVectorStore(str(tmp_path / "chroma"))
    return QdrantVectorStore(":memory:", dimension=DIMENSION)


def query_vector(text: str) -> np.ndarray:
    """Embed a query with the test embedder."""
    return HashingEmbedder(DIMENSION).embed_query(text)


class TestVectorStores:
    def test_add_then_search_finds_the_nearest_chunk(self, store: VectorStore) -> None:
        target = record("attendance rules for examinations")
        store.add_chunks(embedded(target, record("hostel mess menu timings")))
        hits = store.search(query_vector("attendance rules for examinations"), 1)
        assert [hit.chunk_id for hit in hits] == [target.id]
        assert hits[0].score == pytest.approx(1.0, abs=1e-3)

    def test_results_carry_all_metadata(self, store: VectorStore) -> None:
        item = record("some text", department=None, semester=None, section_heading=None)
        store.add_chunks(embedded(item))
        hit = store.search(query_vector("some text"), 1)[0]
        assert hit.text == "some text"
        assert hit.document_id == item.document_id
        assert hit.title == "Doc"
        assert hit.page_number == 3
        assert (hit.department, hit.semester, hit.section_heading) == (None, None, None)
        assert hit.uploaded_at == item.uploaded_at.isoformat()

    def test_missing_page_number_round_trips_as_none(self, store: VectorStore) -> None:
        store.add_chunks(embedded(record("no page", page_number=None)))
        assert store.search(query_vector("no page"), 1)[0].page_number is None

    def test_count(self, store: VectorStore) -> None:
        assert store.count() == 0
        store.add_chunks(embedded(record("a"), record("b")))
        assert store.count() == 2

    def test_adding_the_same_id_replaces_it(self, store: VectorStore) -> None:
        chunk_id = uuid.uuid4()
        store.add_chunks(embedded(record("old", id=chunk_id)))
        store.add_chunks(embedded(record("new", id=chunk_id)))
        assert store.count() == 1

    def test_delete_removes_only_that_document(self, store: VectorStore) -> None:
        doomed = uuid.uuid4()
        store.add_chunks(
            embedded(
                record("a", document_id=doomed),
                record("b", document_id=doomed),
                record("keep"),
            )
        )
        store.delete_document(doomed)
        assert store.count() == 1
        assert store.search(query_vector("keep"), 5)[0].text == "keep"

    def test_department_filter_excludes_other_departments(self, store: VectorStore) -> None:
        store.add_chunks(embedded(record("x", department="CSE"), record("y", department="ECE")))
        hits = store.search(query_vector("x"), 5, SearchFilters(department="ECE"))
        assert [hit.department for hit in hits] == ["ECE"]

    def test_college_wide_documents_pass_department_and_semester_filters(
        self, store: VectorStore
    ) -> None:
        store.add_chunks(
            embedded(
                record("for everyone", department=None, semester=None),
                record("for ece", department="ECE", semester=3),
            )
        )
        hits = store.search(query_vector("x"), 5, SearchFilters(department="CSE", semester=3))
        assert [hit.text for hit in hits] == ["for everyone"]

    def test_semester_filter(self, store: VectorStore) -> None:
        store.add_chunks(embedded(record("x", semester=3), record("y", semester=5)))
        hits = store.search(query_vector("x"), 5, SearchFilters(semester=5))
        assert [hit.semester for hit in hits] == [5]

    def test_doc_type_filter(self, store: VectorStore) -> None:
        store.add_chunks(embedded(record("x", doc_type="notes"), record("y", doc_type="circular")))
        hits = store.search(query_vector("x"), 5, SearchFilters(doc_type="circular"))
        assert [hit.doc_type for hit in hits] == ["circular"]

    def test_filters_combine(self, store: VectorStore) -> None:
        store.add_chunks(
            embedded(
                record("match", department="CSE", doc_type="notes"),
                record("wrong type", department="CSE", doc_type="circular"),
                record("wrong dept", department="ECE", doc_type="notes"),
            )
        )
        filters = SearchFilters(department="CSE", doc_type="notes")
        assert [hit.text for hit in store.search(query_vector("x"), 5, filters)] == ["match"]

    def test_superseded_documents_are_hidden_by_default(self, store: VectorStore) -> None:
        old = uuid.uuid4()
        store.add_chunks(embedded(record("old rule", document_id=old), record("new rule")))
        store.mark_superseded(old)
        assert [hit.text for hit in store.search(query_vector("x"), 5)] == ["new rule"]

    def test_superseded_documents_can_be_requested(self, store: VectorStore) -> None:
        old = uuid.uuid4()
        store.add_chunks(embedded(record("old rule", document_id=old), record("new rule")))
        store.mark_superseded(old)
        hits = store.search(query_vector("x"), 5, SearchFilters(include_superseded=True))
        assert len(hits) == 2

    def test_searching_an_empty_store_returns_nothing(self, store: VectorStore) -> None:
        assert store.search(query_vector("x"), 5) == []


class TestRrf:
    def test_item_in_both_lists_beats_items_in_one(self) -> None:
        a, b, c = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        merged = rrf_merge([[a, b], [c, b]])
        assert merged[0][0] == b

    def test_scores_follow_the_formula(self) -> None:
        a, b = uuid.uuid4(), uuid.uuid4()
        merged = dict(rrf_merge([[a, b], [b]], k=60))
        assert merged[a] == pytest.approx(1 / 61)
        assert merged[b] == pytest.approx(1 / 62 + 1 / 61)

    def test_weights_shift_the_outcome(self) -> None:
        a, b = uuid.uuid4(), uuid.uuid4()
        assert rrf_merge([[a], [b]], weights=[1.0, 2.0])[0][0] == b
        assert rrf_merge([[a], [b]], weights=[2.0, 1.0])[0][0] == a

    def test_a_zero_weight_turns_a_list_off(self) -> None:
        a, b = uuid.uuid4(), uuid.uuid4()
        assert dict(rrf_merge([[a], [b]], weights=[1.0, 0.0]))[b] == 0.0

    def test_empty_lists_merge_to_nothing(self) -> None:
        assert rrf_merge([[], []]) == []

    def test_a_mismatched_weight_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="one weight"):
            rrf_merge([[uuid.uuid4()]], weights=[1.0, 2.0])


class StubTokenizer:
    """Treats each word as one token."""

    def __init__(self) -> None:
        self.vocabulary: list[str] = []

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        tokens = []
        for word in text.split():
            if word not in self.vocabulary:
                self.vocabulary.append(word)
            tokens.append(self.vocabulary.index(word))
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return " ".join(self.vocabulary[token] for token in tokens)


class StubModel:
    """Records what it was asked to encode, and returns a one-hot vector per input."""

    max_seq_length = 12

    def __init__(self) -> None:
        self.tokenizer = StubTokenizer()
        self.encoded: list[str] = []

    def get_embedding_dimension(self) -> int:
        return 4

    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        self.encoded.extend(texts)
        vectors = np.zeros((len(texts), 4), dtype=np.float32)
        for index, _text in enumerate(texts):
            vectors[index, index % 4] = 1.0
        return vectors


class TestEmbeddings:
    def test_windows_cover_every_token_with_overlap(self) -> None:
        tokens = list(range(25))
        windows = split_into_windows(tokens, 10, 3)
        assert windows[0] == list(range(10))
        assert windows[1][:3] == [7, 8, 9]
        assert {t for window in windows for t in window} == set(tokens)
        assert all(len(window) <= 10 for window in windows)

    def test_an_overlap_as_large_as_the_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            split_into_windows(list(range(25)), 10, 10)

    def test_short_text_is_one_window(self) -> None:
        assert split_into_windows([1, 2], 10, 3) == [[1, 2]]

    def test_text_exactly_at_the_limit_is_one_window(self) -> None:
        assert len(split_into_windows(list(range(10)), 10, 3)) == 1

    def test_short_text_reaches_the_model_unchanged(self) -> None:
        model = StubModel()
        SentenceTransformerEmbedder("stub", model=model).embed_documents(["short text"])
        assert model.encoded == ["short text"]

    def test_long_text_is_windowed_so_its_end_is_embedded(self) -> None:
        model = StubModel()
        long_text = " ".join(f"w{n}" for n in range(30))
        vectors = SentenceTransformerEmbedder("stub", model=model).embed_documents([long_text])
        assert len(model.encoded) > 1
        assert "w29" in model.encoded[-1]
        assert vectors.shape == (1, 4)

    def test_averaged_vectors_are_normalised_again(self) -> None:
        long_text = " ".join(f"w{n}" for n in range(30))
        vectors = SentenceTransformerEmbedder("stub", model=StubModel()).embed_documents(
            [long_text, "short"]
        )
        assert np.linalg.norm(vectors, axis=1) == pytest.approx([1.0, 1.0])

    def test_each_input_gets_its_own_row(self) -> None:
        long_text = " ".join(f"w{n}" for n in range(30))
        embedder = SentenceTransformerEmbedder("stub", model=StubModel())
        assert embedder.embed_documents(["a", long_text, "b"]).shape == (3, 4)

    def test_no_input_gives_an_empty_result(self) -> None:
        embedder = SentenceTransformerEmbedder("stub", model=StubModel())
        assert embedder.embed_documents([]).shape == (0, 4)

    def test_only_queries_get_the_bge_prefix(self) -> None:
        model = StubModel()
        embedder = SentenceTransformerEmbedder("stub", model=model)
        embedder.embed_query("when is the exam")
        embedder.embed_documents(["passage"])
        assert model.encoded == [f"{QUERY_PREFIX}when is the exam", "passage"]

    @pytest.mark.model
    def test_real_model_ranks_the_right_passage_first(self) -> None:
        embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", "cpu")
        passages = embedder.embed_documents(
            ["Students need 75 percent attendance to sit exams.", "The canteen serves lunch at 1."]
        )
        scores = passages @ embedder.embed_query("What attendance do I need for examinations?")
        assert scores[0] > scores[1]
        assert np.linalg.norm(passages, axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)

    @pytest.mark.model
    def test_real_model_notices_text_past_its_512_token_limit(self) -> None:
        embedder = SentenceTransformerEmbedder("BAAI/bge-small-en-v1.5", "cpu")
        filler = "The committee met and discussed routine matters of the department. " * 60
        ending = "The library fine is ten rupees per day. " * 30
        vector = embedder.embed_documents([filler + ending])[0]
        truncated = embedder._encode([filler + ending])[0]
        query = embedder.embed_query("What is the library fine per day?")
        assert float(vector @ query) > float(truncated @ query)


def add_ready_document(
    db: Session, text: str, title: str = "Doc", **fields: object
) -> tuple[uuid.UUID, uuid.UUID]:
    """Insert a ready document with one chunk."""
    document = Document(
        title=title,
        filename="a.txt",
        mime_type="text/plain",
        size_bytes=1,
        sha256=uuid.uuid4().hex + uuid.uuid4().hex,
        storage_path="a",
        status="ready",
        **fields,
    )
    chunk = ChunkMeta(chunk_index=0, text=text, token_count=5)
    document.chunks.append(chunk)
    db.add(document)
    db.commit()
    return document.id, chunk.id


class TestBm25:
    def test_tokenizer_keeps_course_codes_whole(self) -> None:
        assert tokenize("CS-301: DBMS") == ["cs", "301", "dbms"]

    def test_finds_exact_keywords(self, db: Session) -> None:
        add_ready_document(db, "The fee for course CS301 is due in July.")
        add_ready_document(db, "Library timings change during vacations.")
        hits = BM25Index(get_session_factory()).search("CS301", 5)
        assert [hit.text for hit in hits] == ["The fee for course CS301 is due in July."]

    def test_no_match_returns_nothing(self, db: Session) -> None:
        add_ready_document(db, "Some unrelated text.")
        assert BM25Index(get_session_factory()).search("quantum", 5) == []

    def test_a_term_in_every_document_still_matches(self, db: Session) -> None:
        add_ready_document(db, "attendance one")
        add_ready_document(db, "attendance two")
        assert len(BM25Index(get_session_factory()).search("attendance", 5)) == 2

    def test_titles_are_searchable(self, db: Session) -> None:
        add_ready_document(db, "Body text only.", title="Scholarship Rules")
        assert len(BM25Index(get_session_factory()).search("scholarship", 5)) == 1

    def test_unfinished_documents_are_ignored(self, db: Session) -> None:
        document_id, _ = add_ready_document(db, "Findable text here.")
        db.get_one(Document, document_id).status = "embedding"
        db.commit()
        assert BM25Index(get_session_factory()).search("findable", 5) == []

    def test_filters_match_the_vector_store_meaning(self, db: Session) -> None:
        add_ready_document(db, "exam for cse", department="CSE", semester=3)
        add_ready_document(db, "exam for ece", department="ECE", semester=3)
        add_ready_document(db, "exam for everyone")
        hits = BM25Index(get_session_factory()).search(
            "exam", 5, SearchFilters(department="CSE", semester=3)
        )
        assert sorted(hit.text for hit in hits) == ["exam for cse", "exam for everyone"]

    def test_superseded_documents_are_hidden_unless_requested(self, db: Session) -> None:
        new_id, _ = add_ready_document(db, "new attendance rule")
        add_ready_document(db, "old attendance rule", superseded_by_id=new_id)
        index = BM25Index(get_session_factory())
        assert [hit.text for hit in index.search("attendance", 5)] == ["new attendance rule"]
        assert len(index.search("attendance", 5, SearchFilters(include_superseded=True))) == 2

    def test_index_refreshes_after_an_upload(self, db: Session) -> None:
        add_ready_document(db, "first document")
        index = BM25Index(get_session_factory())
        assert index.search("second", 5) == []
        add_ready_document(db, "second document")
        assert len(index.search("second", 5)) == 1

    def test_index_refreshes_after_a_delete(self, db: Session) -> None:
        document_id, _ = add_ready_document(db, "doomed text")
        index = BM25Index(get_session_factory())
        assert len(index.search("doomed", 5)) == 1
        db.delete(db.get_one(Document, document_id))
        db.commit()
        assert index.search("doomed", 5) == []

    def test_empty_corpus_and_empty_query_are_safe(self, db: Session) -> None:
        index = BM25Index(get_session_factory())
        assert index.search("anything", 5) == []
        add_ready_document(db, "text")
        assert index.search("?!", 5) == []


class ReverseReranker:
    """Puts candidates in reverse order, with a fixed score."""

    def rerank(self, query: str, chunks: list[ScoredChunk], top_k: int) -> list[ScoredChunk]:
        return list(reversed(chunks))[:top_k]


@pytest.fixture
def retrieval(db: Session, tmp_path: Path) -> Callable[..., HybridRetriever]:
    """Build a retriever over a real Chroma store and real BM25, with hashed embeddings."""
    store = ChromaVectorStore(str(tmp_path / "chroma"))

    def build(**options: object) -> HybridRetriever:
        embedder = HashingEmbedder(DIMENSION)
        return HybridRetriever(store, embedder, BM25Index(get_session_factory()), **options)  # type: ignore[arg-type]

    build.store = store  # type: ignore[attr-defined]
    return build


def add_to_both(db: Session, store: VectorStore, text: str, **fields: object) -> uuid.UUID:
    """Add a document to Postgres and the vector store with matching ids."""
    document_id, chunk_id = add_ready_document(db, text, **fields)
    store.add_chunks(embedded(record(text, id=chunk_id, document_id=document_id)))
    return chunk_id


class TestHybridRetriever:
    def test_each_mode_returns_results(
        self, db: Session, retrieval: Callable[..., HybridRetriever]
    ) -> None:
        add_to_both(db, retrieval.store, "attendance rules for exams")  # type: ignore[attr-defined]
        retriever = retrieval(reranker=ReverseReranker())
        for mode in SearchMode:
            assert retriever.retrieve("attendance rules", mode=mode).chunks

    def test_hybrid_finds_a_keyword_vector_search_ranks_low(
        self, db: Session, retrieval: Callable[..., HybridRetriever]
    ) -> None:
        for index in range(30):
            add_to_both(db, retrieval.store, f"general campus information item {index}")  # type: ignore[attr-defined]
        target = add_to_both(db, retrieval.store, "CS301 lab manual")  # type: ignore[attr-defined]
        result = retrieval().retrieve("CS301", k=5, mode=SearchMode.HYBRID)
        assert target in [chunk.chunk_id for chunk in result.chunks]

    def test_chunk_in_both_searches_ranks_first(
        self, db: Session, retrieval: Callable[..., HybridRetriever]
    ) -> None:
        target = add_to_both(db, retrieval.store, "attendance rules")  # type: ignore[attr-defined]
        add_to_both(db, retrieval.store, "unrelated hostel menu")  # type: ignore[attr-defined]
        assert retrieval().retrieve("attendance rules").chunks[0].chunk_id == target

    def test_result_reports_cosine_similarity_alongside_rrf_scores(
        self, db: Session, retrieval: Callable[..., HybridRetriever]
    ) -> None:
        target = add_to_both(db, retrieval.store, "attendance rules")  # type: ignore[attr-defined]
        result: RetrievalResult = retrieval().retrieve("attendance rules")
        assert result.similarities[target] == pytest.approx(1.0, abs=1e-3)
        assert result.chunks[0].score < 0.1

    def test_filters_reach_both_searches(
        self, db: Session, retrieval: Callable[..., HybridRetriever]
    ) -> None:
        store = retrieval.store  # type: ignore[attr-defined]
        document_id, chunk_id = add_ready_document(db, "exam schedule", department="ECE")
        store.add_chunks(
            embedded(
                record("exam schedule", id=chunk_id, document_id=document_id, department="ECE")
            )
        )
        result = retrieval().retrieve("exam schedule", filters=SearchFilters(department="CSE"))
        assert result.chunks == []

    def test_reranker_decides_the_final_order(
        self, db: Session, retrieval: Callable[..., HybridRetriever]
    ) -> None:
        first = add_to_both(db, retrieval.store, "attendance rules")  # type: ignore[attr-defined]
        add_to_both(db, retrieval.store, "attendance")  # type: ignore[attr-defined]
        retriever = retrieval(reranker=ReverseReranker())
        plain = retriever.retrieve("attendance rules", mode=SearchMode.HYBRID)
        reranked = retriever.retrieve("attendance rules", mode=SearchMode.HYBRID_RERANK)
        assert plain.chunks[0].chunk_id == first
        assert reranked.chunks[-1].chunk_id == first

    def test_rerank_mode_without_a_reranker_is_an_error(
        self, db: Session, retrieval: Callable[..., HybridRetriever]
    ) -> None:
        with pytest.raises(ValueError, match="re-ranker"):
            retrieval().retrieve("x", mode=SearchMode.HYBRID_RERANK)


class TestIndexer:
    def test_embedding_text_includes_title_and_heading(self) -> None:
        text = embedding_text(record("Body.", title="Rules", section_heading="Unit 1"))
        assert text == "Rules\nUnit 1\nBody."

    def test_pipeline_output_is_searchable(
        self, deps: PipelineDeps, db: Session, tmp_path: Path
    ) -> None:
        store = ChromaVectorStore(str(tmp_path / "chroma"))
        deps.indexer = VectorIndexer(store, HashingEmbedder(DIMENSION))
        text = b"Students need seventy five percent attendance to sit the final examinations."
        document = create_document(
            db, deps.storage, filename="rules.txt", data=text, uploader_id=None, doc_type="notice"
        ).document
        db.commit()
        assert run_ingestion(deps, document.id) is True
        hits = store.search(query_vector("attendance"), 3)
        assert hits[0].document_id == document.id

    def test_superseding_hides_the_old_version_end_to_end(
        self, deps: PipelineDeps, db: Session, tmp_path: Path
    ) -> None:
        store = ChromaVectorStore(str(tmp_path / "chroma"))
        deps.indexer = VectorIndexer(store, HashingEmbedder(DIMENSION))
        body = " ".join(f"Clause {n} sets out the examination procedure." for n in range(50))
        ids = []
        for name, text in (("v1.txt", body), ("v2.txt", body + " Amended.")):
            document = create_document(
                db,
                deps.storage,
                filename=name,
                data=text.encode(),
                uploader_id=None,
                doc_type="circular",
            ).document
            db.commit()
            run_ingestion(deps, document.id)
            ids.append(document.id)
        hits = store.search(query_vector("examination procedure"), 10)
        assert {hit.document_id for hit in hits} == {ids[1]}

    def test_progress_is_reported_per_batch(self) -> None:
        class CountingStore(ChromaVectorStore):
            def __init__(self) -> None:
                self.batches: list[int] = []

            def add_chunks(self, chunks: object) -> None:
                self.batches.append(len(chunks))  # type: ignore[arg-type]

        store = CountingStore()
        progress: list[tuple[int, int]] = []
        indexer = VectorIndexer(store, HashingEmbedder(DIMENSION))
        indexer.index([record(str(n)) for n in range(40)], lambda *pair: progress.append(pair))
        assert store.batches == [32, 8]
        assert progress == [(32, 40), (40, 40)]


def load_benchmark():
    """Import eval/retrieval_bench.py, which lives outside the backend package."""
    path = Path(__file__).resolve().parents[2] / "eval" / "retrieval_bench.py"
    spec = importlib.util.spec_from_file_location("retrieval_bench", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["retrieval_bench"] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


class TestBenchmarkMetrics:
    bench = load_benchmark()

    def test_perfect_retrieval(self) -> None:
        scores = self.bench.score_runs([(["a"], {"a"})])
        assert (scores.precision, scores.recall, scores.mrr) == (1.0, 1.0, 1.0)

    def test_repeated_documents_count_once(self) -> None:
        assert self.bench.precision_at_k(["a", "a", "b"], {"a"}) == 0.5

    def test_recall_needs_every_relevant_document(self) -> None:
        assert self.bench.recall_at_k(["a"], {"a", "b"}) == 0.5

    def test_mrr_uses_the_first_relevant_rank(self) -> None:
        assert self.bench.reciprocal_rank(["x", "x", "y", "a"], {"a"}) == pytest.approx(1 / 3)

    def test_mrr_is_zero_when_nothing_relevant_was_found(self) -> None:
        assert self.bench.reciprocal_rank(["x"], {"a"}) == 0.0

    def test_scores_are_averaged_over_questions(self) -> None:
        scores = self.bench.score_runs([(["a"], {"a"}), (["b"], {"a"})])
        assert scores.recall == 0.5

    def test_no_questions_gives_zero(self) -> None:
        assert self.bench.score_runs([]).mrr == 0.0

    def test_unanswerable_questions_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "q.json"
        path.write_text(
            '[{"question": "a", "relevant_document_ids": []},'
            ' {"question": "b", "relevant_document_ids": ["x"]}]'
        )
        assert [item["question"] for item in self.bench.load_questions(path)] == ["b"]

    def test_benchmark_runs_every_mode(
        self, db: Session, retrieval: Callable[..., HybridRetriever]
    ) -> None:
        document_id, chunk_id = add_ready_document(db, "attendance rules")
        retrieval.store.add_chunks(  # type: ignore[attr-defined]
            embedded(record("attendance rules", id=chunk_id, document_id=document_id))
        )
        questions = [{"question": "attendance", "relevant_document_ids": [str(document_id)]}]
        results = self.bench.benchmark(
            retrieval(reranker=ReverseReranker()), questions, 5, self.bench.MODES
        )
        assert set(results) == set(self.bench.MODES)
        assert all(scores.recall == 1.0 for scores in results.values())
