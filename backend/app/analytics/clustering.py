"""Grouping similar questions by embedding similarity."""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from app.vectorstore.embeddings import Embedder


@dataclass
class Cluster:
    """Questions that mean roughly the same thing."""

    representative: str
    count: int = 0
    examples: list[str] = field(default_factory=list)


def _normalize(question: str) -> str:
    """Treat questions differing only in case or spacing as identical."""
    return " ".join(question.lower().split())


def cluster_questions(
    questions: list[str], embedder: Embedder, threshold: float = 0.85, max_examples: int = 3
) -> list[Cluster]:
    """Group questions, largest group first.

    Each distinct question joins the first group whose leader is similar enough, and
    otherwise starts a group of its own. Asking in frequency order means the commonest
    wording becomes the leader, and using a fixed leader, not a moving average, stops a
    group drifting away from what it started as. Vectors are normalised, so a dot product
    is the cosine similarity.
    """
    counts: Counter[str] = Counter()
    original: dict[str, str] = {}
    for question in questions:
        key = _normalize(question)
        if key:
            counts[key] += 1
            original.setdefault(key, question.strip())
    if not counts:
        return []

    ordered = [key for key, _ in counts.most_common()]
    vectors = embedder.embed_documents([original[key] for key in ordered])

    clusters: list[Cluster] = []
    leaders: list[np.ndarray] = []
    for key, vector in zip(ordered, vectors, strict=True):
        text = original[key]
        best = -1
        if leaders:
            similarities = np.stack(leaders) @ vector
            if float(similarities.max()) > threshold:
                best = int(similarities.argmax())
        if best < 0:
            clusters.append(Cluster(representative=text))
            leaders.append(vector)
            best = len(clusters) - 1
        cluster = clusters[best]
        cluster.count += counts[key]
        if len(cluster.examples) < max_examples:
            cluster.examples.append(text)
    return sorted(clusters, key=lambda item: (-item.count, item.representative))
