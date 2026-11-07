"""Turn drafted questions into a spreadsheet for a person to check, and read it back.

Usage (from the repository root):

    python eval/review_sheet.py export eval/data/testset_draft.json eval/data/review.csv \
        --ids eval/corpus/lucknow.ids.json
    python eval/review_sheet.py import eval/data/review.csv eval/data/testset.json \
        --ids eval/corpus/lucknow.ids.json

Open the CSV in Excel or Google Sheets. For each row, check the answer at the place named in
`source`, correct the question or answer if needed, and put `y` in `keep`. Rows without `y`
are dropped. Add your own rows, such as multi-hop questions, with the category and the file
names of every document needed, separated by semicolons. Only kept rows are written, and
each is marked verified.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from dataset import LabelledQuestion, load_testset, save_testset, summarize

COLUMNS = ["n", "keep", "category", "question", "gold_answer", "source", "documents"]


def export_sheet(questions: list[LabelledQuestion], ids: dict[str, str], path: Path) -> None:
    """Write questions as rows, naming documents by file rather than by database id."""
    files = {document_id: name for name, document_id in ids.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig so Excel shows rupee signs and dashes correctly.
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for number, question in enumerate(questions, start=1):
            names = [
                files.get(document_id, document_id)
                for document_id in question.relevant_document_ids
            ]
            writer.writerow(
                [
                    number,
                    "",
                    question.category,
                    question.question,
                    question.gold_answer,
                    question.source,
                    "; ".join(names),
                ]
            )


def import_sheet(path: Path, ids: dict[str, str]) -> list[LabelledQuestion]:
    """Read back the rows marked keep, as verified questions."""
    kept: list[LabelledQuestion] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (row.get("keep") or "").strip().lower() not in {"y", "yes"}:
                continue
            names = [
                name.strip() for name in (row.get("documents") or "").split(";") if name.strip()
            ]
            unknown = [name for name in names if name not in ids]
            if unknown:
                raise ValueError(
                    f"Row {row.get('n') or '(new)'}: unknown document {', '.join(unknown)}"
                )
            kept.append(
                LabelledQuestion(
                    question=row["question"].strip(),
                    gold_answer=row["gold_answer"].strip(),
                    relevant_document_ids=[ids[name] for name in names],
                    category=row["category"].strip(),
                    source=(row.get("source") or "").strip(),
                    verified=True,
                )
            )
    return kept


def main() -> int:
    """Export or import a review sheet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["export", "import"])
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--ids", type=Path, required=True, help="the corpus ids file")
    args = parser.parse_args()
    ids = json.loads(args.ids.read_text(encoding="utf-8"))

    if args.command == "export":
        questions = load_testset(args.source, require_verified=False)
        export_sheet(questions, ids, args.target)
        print(f"Wrote {len(questions)} rows to {args.target}")
        return 0
    try:
        kept = import_sheet(args.source, ids)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    save_testset(kept, args.target)
    print(f"Kept {len(kept)} verified questions: {summarize(kept)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
