"""Compare retrieval strategies on a labelled question set.

Usage (from backend/, with DATABASE_URL and the vector store configured):

    python ../eval/retrieval_bench.py ../eval/data/questions.json --output ../eval/results

The input is a JSON list of {"question": "...", "relevant_document_ids": ["...", ...]}.
Questions with no relevant documents are skipped, because they have nothing to find.
"""

import argparse
import csv
import json
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class Scores:
    """Average retrieval quality for one strategy."""

    precision: float
    recall: float
    mrr: float


def unique_in_order(items: list[str]) -> list[str]:
    """Remove repeats, keeping first positions. Several chunks can share a document."""
    return list(dict.fromkeys(items))


def precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Return the share of retrieved documents that are relevant."""
    documents = unique_in_order(retrieved)
    return len(relevant.intersection(documents)) / len(documents) if documents else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    """Return the share of relevant documents that were retrieved."""
    return len(relevant.intersection(retrieved)) / len(relevant) if relevant else 0.0


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    """Return 1 / rank of the first relevant document, or 0 if none was retrieved."""
    for rank, document in enumerate(unique_in_order(retrieved), start=1):
        if document in relevant:
            return 1.0 / rank
    return 0.0


def score_runs(runs: list[tuple[list[str], set[str]]]) -> Scores:
    """Average the metrics over (retrieved document ids, relevant ids) pairs."""
    if not runs:
        return Scores(0.0, 0.0, 0.0)
    count = len(runs)
    return Scores(
        precision=sum(precision_at_k(*run) for run in runs) / count,
        recall=sum(recall_at_k(*run) for run in runs) / count,
        mrr=sum(reciprocal_rank(*run) for run in runs) / count,
    )


def load_questions(path: Path) -> list[dict]:
    """Read the labelled questions, skipping any that have no relevant documents."""
    questions = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in questions if item.get("relevant_document_ids")]


def benchmark(
    retriever, questions: list[dict], k: int, modes: tuple[str, ...]
) -> dict[str, Scores]:
    """Run every question through every mode."""
    from app.vectorstore import SearchMode

    results = {}
    for mode in modes:
        runs = []
        for item in questions:
            hits = retriever.retrieve(item["question"], k=k, mode=SearchMode(mode)).chunks
            retrieved = [str(chunk.document_id) for chunk in hits]
            runs.append((retrieved, {str(uuid.UUID(i)) for i in item["relevant_document_ids"]}))
        results[mode] = score_runs(runs)
    return results


def write_results(results: dict[str, Scores], output: Path, k: int) -> None:
    """Save the results as CSV and print a Markdown table."""
    output.mkdir(parents=True, exist_ok=True)
    with (output / "retrieval.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", f"precision@{k}", f"recall@{k}", "mrr"])
        for mode, scores in results.items():
            writer.writerow(
                [mode, f"{scores.precision:.4f}", f"{scores.recall:.4f}", f"{scores.mrr:.4f}"]
            )

    print(f"| Mode | Precision@{k} | Recall@{k} | MRR |")
    print("|---|---|---|---|")
    for mode, scores in results.items():
        print(f"| {mode} | {scores.precision:.3f} | {scores.recall:.3f} | {scores.mrr:.3f} |")


def main() -> None:
    """Run the benchmark from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("../eval/results"))
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    args = parser.parse_args()

    from app.config import get_settings
    from app.vectorstore import get_retriever

    if "hybrid_rerank" in args.modes and not get_settings().rerank_enabled:
        raise SystemExit("Set RERANK_ENABLED=true to include hybrid_rerank.")

    questions = load_questions(args.questions)
    results = benchmark(get_retriever(), questions, args.k, tuple(args.modes))
    write_results(results, args.output, args.k)


if __name__ == "__main__":
    main()
