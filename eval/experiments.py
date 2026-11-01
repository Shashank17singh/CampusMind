"""Run the retrieval experiments and tabulate them.

Usage (from backend/, against a database used only for evaluation):

    python ../eval/experiments.py ../eval/data/testset.json --output ../eval/results

Chunk size and embedding model need the corpus rebuilt, so each configuration re-ingests
every document into its own vector store folder and rewrites `chunks_meta`. Never point
this at a production database. Relevance is judged per document, so document ids stay valid
across chunk sizes.

To compare hosted and local LLMs, run `run_eval.py` once per LLM_PROVIDER.
"""

import argparse
import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_testset  # noqa: E402
from metrics import paired_bootstrap  # noqa: E402
from report import write_report  # noqa: E402
from retrieval_bench import reciprocal_rank, score_runs  # noqa: E402

BGE_SMALL = "BAAI/bge-small-en-v1.5"
MINILM = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class Config:
    """One experimental setting. Each varies a single factor from the baseline."""

    experiment: str
    name: str
    chunk_size: int = 800
    embedding_model: str = BGE_SMALL
    mode: str = "hybrid"


def default_grid() -> list[Config]:
    """Return every configuration to test, each compared against the same baseline."""
    return [
        Config("chunk_size", "400", chunk_size=400),
        Config("chunk_size", "800"),
        Config("chunk_size", "1200", chunk_size=1200),
        Config("embedding", "bge-small"),
        Config("embedding", "all-MiniLM-L6-v2", embedding_model=MINILM),
        Config("retrieval", "vector", mode="vector"),
        Config("retrieval", "hybrid"),
        Config("retrieval", "hybrid+rerank", mode="hybrid_rerank"),
    ]


def corpus_key(config: Config) -> tuple[int, str]:
    """Identify the settings that decide how a corpus is built."""
    return (config.chunk_size, config.embedding_model)


def build_corpus(config: Config, work_dir: Path) -> Any:
    """Re-ingest every document under this configuration, and return a retriever."""
    from sqlalchemy import select

    from app.config import get_settings
    from app.db import get_session_factory
    from app.ingestion.ocr import TesseractOcr
    from app.ingestion.pipeline import PipelineDeps, run_ingestion
    from app.ingestion.storage import LocalStorage
    from app.llm import get_llm
    from app.models import Document
    from app.vectorstore.bm25 import BM25Index
    from app.vectorstore.chroma_store import ChromaVectorStore
    from app.vectorstore.embeddings import QUERY_PREFIX, SentenceTransformerEmbedder
    from app.vectorstore.hybrid import HybridRetriever
    from app.vectorstore.indexer import VectorIndexer
    from app.vectorstore.rerank import CrossEncoderReranker

    settings = get_settings().model_copy(
        update={
            "chunk_size_tokens": config.chunk_size,
            "chunk_overlap_tokens": config.chunk_size * 15 // 100,
        }
    )
    prefix = QUERY_PREFIX if "bge" in config.embedding_model else ""
    embedder = SentenceTransformerEmbedder(config.embedding_model, query_prefix=prefix)
    folder = work_dir / f"{config.chunk_size}-{config.embedding_model.replace('/', '_')}"
    store = ChromaVectorStore(str(folder))
    factory = get_session_factory()
    deps = PipelineDeps(
        session_factory=factory,
        storage=LocalStorage(settings.upload_dir),
        ocr=TesseractOcr(settings.tesseract_cmd),
        llm=get_llm(),
        indexer=VectorIndexer(store, embedder),
        settings=settings,
    )
    with factory() as session:
        documents = session.scalars(select(Document)).all()
        for document in documents:
            document.status = "uploaded"
        session.commit()
        identifiers = [document.id for document in documents]
    for document_id in identifiers:
        run_ingestion(deps, document_id)

    reranker = CrossEncoderReranker(settings.reranker_model) if settings.rerank_enabled else None
    return HybridRetriever(store, embedder, BM25Index(factory), reranker=reranker)


def score_config(retriever: Any, questions: list[Any], mode: str, k: int = 5) -> dict[str, Any]:
    """Score one configuration, keeping per-question values for significance testing."""
    from app.vectorstore import SearchMode

    runs = []
    for item in questions:
        hits = retriever.retrieve(item.question, k=k, mode=SearchMode(mode)).chunks
        runs.append(([str(chunk.document_id) for chunk in hits], set(item.relevant_document_ids)))
    scores = score_runs(runs)
    return {
        "precision": scores.precision,
        "recall": scores.recall,
        "mrr": scores.mrr,
        "per_question_rr": [reciprocal_rank(*run) for run in runs],
    }


def run_grid(grid: list[Config], questions: list[Any], work_dir: Path) -> list[dict[str, Any]]:
    """Run every configuration, building each distinct corpus once."""
    answerable = [item for item in questions if item.relevant_document_ids]
    retrievers: dict[tuple[int, str], Any] = {}
    rows = []
    for config in grid:
        key = corpus_key(config)
        if key not in retrievers:
            print(f"Building corpus {key}", flush=True)
            retrievers[key] = build_corpus(config, work_dir)
        scores = score_config(retrievers[key], answerable, config.mode)
        rows.append({"experiment": config.experiment, "name": config.name, **scores})
        print(f"{config.experiment}/{config.name}: {scores['recall']:.3f} recall", flush=True)
    return rows


def add_significance(rows: list[dict[str, Any]], baseline: dict[str, str]) -> None:
    """Compare each row with its experiment's baseline, on per-question reciprocal rank."""
    for row in rows:
        base_name = baseline.get(row["experiment"])
        reference = next(
            (r for r in rows if r["experiment"] == row["experiment"] and r["name"] == base_name),
            None,
        )
        if reference is None or reference is row:
            row["mrr_difference"] = row["ci_low"] = row["ci_high"] = None
            continue
        difference, low, high = paired_bootstrap(
            reference["per_question_rr"], row["per_question_rr"]
        )
        row["mrr_difference"], row["ci_low"], row["ci_high"] = difference, low, high


def main() -> None:
    """Run the experiment grid from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("testset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("../eval/results"))
    parser.add_argument("--allow-unverified", action="store_true")
    parser.add_argument("--skip-rerank", action="store_true")
    args = parser.parse_args()

    questions = load_testset(args.testset, require_verified=not args.allow_unverified)
    synthetic = args.allow_unverified or any(item.synthetic for item in questions)
    grid = default_grid()
    if args.skip_rerank:
        grid = [config for config in grid if config.mode != "hybrid_rerank"]

    with tempfile.TemporaryDirectory(prefix="campusmind-eval-", ignore_cleanup_errors=True) as work:
        rows = run_grid(grid, questions, Path(work))
    add_significance(rows, {"chunk_size": "800", "embedding": "bge-small", "retrieval": "hybrid"})

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "experiments.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_report(rows, args.output, synthetic=synthetic, questions=len(questions))


if __name__ == "__main__":
    main()
