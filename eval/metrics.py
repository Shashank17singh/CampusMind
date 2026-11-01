"""Metrics that do not depend on the application: agreement, percentiles and significance."""

import math
import random
from collections import Counter
from collections.abc import Sequence


def percentile(values: Sequence[float], rank: float) -> float:
    """Return a percentile with linear interpolation. `rank` runs from 0 to 100."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * rank / 100
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def cohen_kappa(first: Sequence[int], second: Sequence[int]) -> float:
    """Measure agreement between two raters, corrected for agreement by chance.

    Plain percentage agreement flatters a judge: if nearly every answer is rated 5, two
    raters agree most of the time by accident. Kappa is 1 for perfect agreement and 0 for
    no better than chance.
    """
    if len(first) != len(second):
        raise ValueError("Both raters must score the same items")
    if not first:
        return 0.0
    total = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / total
    first_counts, second_counts = Counter(first), Counter(second)
    expected = sum(first_counts[label] * second_counts[label] for label in first_counts) / (
        total * total
    )
    if expected == 1:
        return 1.0
    return (observed - expected) / (1 - expected)


def paired_bootstrap(
    baseline: Sequence[float],
    candidate: Sequence[float],
    samples: int = 10_000,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Return the mean difference and its 95% interval, from per-question scores.

    If the interval contains 0, the sample cannot show that the candidate is better. The
    pairing matters: both systems answered the same questions, so the comparison is
    question by question, not of two separate averages.
    """
    if len(baseline) != len(candidate):
        raise ValueError("Both systems must answer the same questions")
    if not baseline:
        return (0.0, 0.0, 0.0)
    differences = [b - a for a, b in zip(baseline, candidate, strict=True)]
    generator = random.Random(seed)  # noqa: S311
    count = len(differences)
    means = sorted(sum(generator.choices(differences, k=count)) / count for _ in range(samples))
    return (sum(differences) / count, percentile(means, 2.5), percentile(means, 97.5))
