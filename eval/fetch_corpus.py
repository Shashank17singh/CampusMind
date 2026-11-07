"""Download an evaluation corpus listed in a manifest, and check every file's checksum.

Usage (from the repository root):

    python eval/fetch_corpus.py eval/corpus/mmmut.json

Files land beside the manifest, in a folder with the manifest's name (eval/corpus/mmmut/),
which git ignores. A file that is already present with the right checksum is not downloaded
again. A changed file is reported, not used: the university may have replaced it, and
questions written against the old version would no longer hold.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

USER_AGENT = "CampusMind evaluation corpus fetcher"


def sha256(path: Path) -> str:
    """The file's SHA-256, in hex."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(manifest_path: Path) -> int:
    """Download every document. Returns the number of problems."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    folder = manifest_path.with_suffix("")
    folder.mkdir(exist_ok=True)
    problems = 0
    with httpx.Client(
        timeout=120, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        for entry in manifest["documents"]:
            target = folder / entry["file"]
            if target.exists() and sha256(target) == entry["sha256"]:
                print(f"present   {entry['file']}")
                continue
            try:
                response = client.get(entry["url"])
                response.raise_for_status()
            except httpx.HTTPError as exc:
                print(f"FAILED    {entry['file']}: {exc}", file=sys.stderr)
                problems += 1
                continue
            digest = hashlib.sha256(response.content).hexdigest()
            if digest != entry["sha256"]:
                print(
                    f"CHANGED   {entry['file']}: the source file differs from the manifest",
                    file=sys.stderr,
                )
                problems += 1
                continue
            target.write_bytes(response.content)
            print(f"fetched   {entry['file']}")
    return problems


def main() -> int:
    """Fetch the corpus named on the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    problems = fetch(args.manifest)
    if problems:
        print(f"{problems} document(s) could not be used.", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
