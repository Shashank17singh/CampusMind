"""Write experiment results as CSV, Markdown tables and charts for the paper.

Charts are built for greyscale print: each metric has its own shade and hatch, one axis, a
legend, a value on each bar, and a y axis from 0 to 1, so differences look the size they are.
"""

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

METRICS = ("precision", "recall", "mrr")
LABELS = {"precision": "Precision@5", "recall": "Recall@5", "mrr": "MRR"}
TITLES = {
    "chunk_size": "Chunk size (tokens)",
    "embedding": "Embedding model",
    "retrieval": "Retrieval strategy",
}
STYLE = {
    "precision": {"color": "#ffffff", "hatch": "///"},
    "recall": {"color": "#8c8c8c", "hatch": ""},
    "mrr": {"color": "#262626", "hatch": ""},
}
SYNTHETIC_NOTE = (
    "Synthetic corpus and unverified questions. This shows the tooling works. "
    "These numbers are not results and must not be reported."
)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Save the results table, without the per-question detail."""
    fields = ["experiment", "name", *METRICS, "mrr_difference", "ci_low", "ci_high"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _difference(row: dict[str, Any]) -> str:
    """Show the change in MRR from the baseline, and whether it is distinguishable from 0."""
    if row.get("mrr_difference") is None:
        return "baseline"
    interval = f"{row['mrr_difference']:+.3f} [{row['ci_low']:+.3f}, {row['ci_high']:+.3f}]"
    return f"{interval} (no clear difference)" if row["ci_low"] <= 0 <= row["ci_high"] else interval


def markdown_tables(rows: list[dict[str, Any]], synthetic: bool, questions: int) -> str:
    """Build one table per experiment."""
    lines = ["# Retrieval experiments", "", f"Questions: {questions}.", ""]
    if synthetic:
        lines += [f"**{SYNTHETIC_NOTE}**", ""]
    for experiment, title in TITLES.items():
        group = [row for row in rows if row["experiment"] == experiment]
        if not group:
            continue
        lines += [
            f"## {title}",
            "",
            "| Setting | Precision@5 | Recall@5 | MRR | MRR change, 95% interval |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {row['name']} | {row['precision']:.3f} | {row['recall']:.3f} "
            f"| {row['mrr']:.3f} | {_difference(row)} |"
            for row in group
        ]
        lines.append("")
    return "\n".join(lines)


def bar_chart(group: list[dict[str, Any]], title: str, path: Path, synthetic: bool) -> None:
    """Draw one grouped bar chart."""
    plt.rcParams.update({"font.family": "serif", "font.size": 9, "hatch.linewidth": 0.5})
    figure, axis = plt.subplots(figsize=(3.5, 2.6))
    width = 0.8 / len(METRICS)
    for index, metric in enumerate(METRICS):
        positions = [slot + (index - 1) * width for slot in range(len(group))]
        bars = axis.bar(
            positions,
            [row[metric] for row in group],
            width * 0.92,
            label=LABELS[metric],
            edgecolor="#000000",
            linewidth=0.5,
            **STYLE[metric],
        )
        axis.bar_label(bars, fmt="%.2f", fontsize=6, padding=1)
    axis.set_xticks(range(len(group)), [row["name"] for row in group])
    axis.set_xlabel(title)
    axis.set_ylim(0, 1.15)
    axis.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncols=3, fontsize=7, loc="upper center")
    if synthetic:
        figure.text(0.5, 0.5, "SYNTHETIC", ha="center", va="center", fontsize=28, alpha=0.12)
    figure.tight_layout()
    figure.savefig(path.with_suffix(".png"), dpi=300)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def write_report(
    rows: list[dict[str, Any]], output: Path, synthetic: bool = False, questions: int = 0
) -> None:
    """Write the CSV, tables and charts."""
    output.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output / "experiments.csv")
    (output / "experiments.md").write_text(
        markdown_tables(rows, synthetic, questions), encoding="utf-8"
    )
    for experiment, title in TITLES.items():
        group = [row for row in rows if row["experiment"] == experiment]
        if group:
            bar_chart(group, title, output / f"{experiment}", synthetic)
