"""Ingest a fetched evaluation corpus, using the metadata in its manifest.

Usage (from backend/, against a database used only for evaluation):

    python ../eval/ingest_corpus.py ../eval/corpus/mmmut.json

Each document goes through the real pipeline, OCR included, with the title, type and
department from the manifest rather than the auto-tagger, so tagging mistakes cannot leak
into retrieval scores. The document ids are written to `<manifest>.ids.json`, which the
test set uses for relevance. They differ between databases, so that file is not committed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))


def ingest(manifest_path: Path) -> int:
    """Ingest every document. Returns the number that failed."""
    from app.config import get_settings
    from app.db import get_session_factory
    from app.ingestion.pipeline import run_ingestion
    from app.ingestion.service import create_document
    from app.ingestion.tasks import build_deps

    if get_settings().environment == "production":
        print("Refusing to ingest an evaluation corpus into production.", file=sys.stderr)
        return 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folder = manifest_path.with_suffix("")
    deps = build_deps()
    ids: dict[str, str] = {}
    failed = 0
    with get_session_factory()() as session:
        for entry in manifest["documents"]:
            path = folder / entry["file"]
            if not path.exists():
                print(f"MISSING   {entry['file']}: run fetch_corpus.py first", file=sys.stderr)
                failed += 1
                continue
            result = create_document(
                session,
                deps.storage,
                filename=entry["file"],
                data=path.read_bytes(),
                uploader_id=None,
                title=entry["title"],
                doc_type=entry["doc_type"],
                department=entry["department"],
                semester=entry.get("semester"),
            )
            session.commit()
            ids[entry["file"]] = str(result.document.id)
            if result.duplicate:
                print(f"present   {entry['file']}")
                continue
            started = time.perf_counter()
            ok = run_ingestion(deps, result.document.id)
            seconds = time.perf_counter() - started
            print(f"{'ingested' if ok else 'FAILED  '}  {entry['file']} ({seconds:.0f}s)")
            failed += 0 if ok else 1

    ids_path = manifest_path.with_suffix(".ids.json")
    ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    print(f"Document ids written to {ids_path}")
    return failed


def main() -> int:
    """Ingest the corpus named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    return 1 if ingest(args.manifest) else 0


if __name__ == "__main__":
    raise SystemExit(main())
