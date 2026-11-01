"""Create demo users and 20 fictional documents. Safe to run more than once.

Usage: SEED_PASSWORD='choose-one' python scripts/seed.py [--testset path.json]

The password comes from the environment, so no working credential is stored in the repo.
Never run this against a production database.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sample_documents import SAMPLES  # noqa: E402

DEMO_USERS = (
    ("admin", "Demo Admin", None, None),
    ("faculty", "Demo Faculty", "CSE", None),
    ("student", "Demo Student", "CSE", 3),
)


def seed_users(session, password: str) -> None:
    """Create one verified account per role."""
    from app.auth.passwords import hash_password
    from app.auth.service import find_user
    from app.config import get_settings
    from app.models import User
    from app.models.base import utcnow

    domain = sorted(get_settings().allowed_email_domains)[0]
    hashed = hash_password(password)
    for role, name, department, semester in DEMO_USERS:
        email = f"{role}@{domain}"
        if find_user(session, email) is None:
            session.add(
                User(
                    email=email,
                    full_name=name,
                    role=role,
                    department=department,
                    semester=semester,
                    password_hash=hashed,
                    email_verified_at=utcnow(),
                )
            )
            print(f"Created {email}")


def seed_documents(session, deps) -> dict[str, str]:
    """Store and ingest the sample documents. Returns filename to document id."""
    from app.ingestion.pipeline import run_ingestion
    from app.ingestion.service import create_document

    identifiers: dict[str, str] = {}
    for sample in SAMPLES:
        result = create_document(
            session,
            deps.storage,
            filename=sample.filename,
            data=sample.text.encode(),
            uploader_id=None,
            title=sample.title,
            doc_type=sample.doc_type,
            department=sample.department,
            semester=sample.semester,
        )
        session.commit()
        identifiers[sample.filename] = str(result.document.id)
        if not result.duplicate:
            ok = run_ingestion(deps, result.document.id)
            print(f"{'Ingested' if ok else 'FAILED'}: {sample.title}")
    return identifiers


def write_testset(identifiers: dict[str, str], path: Path) -> None:
    """Write questions for the samples. They are synthetic and stay unverified."""
    from dataset import LabelledQuestion, save_testset
    from draft_testset import unanswerable_questions

    questions = [
        LabelledQuestion(
            question=question,
            gold_answer=answer,
            relevant_document_ids=[identifiers[sample.filename]],
            category="factual",
            synthetic=True,
        )
        for sample in SAMPLES
        for question, answer in sample.questions
    ]
    questions += [case.model_copy(update={"synthetic": True}) for case in unanswerable_questions()]
    save_testset(questions, path)
    print(f"Wrote {len(questions)} synthetic questions to {path}")


def main() -> int:
    """Seed the database."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", type=Path)
    args = parser.parse_args()

    password = os.environ.get("SEED_PASSWORD", "")
    if len(password) < 10:
        print("Set SEED_PASSWORD to at least 10 characters.", file=sys.stderr)
        return 1

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
    from app.config import get_settings
    from app.db import get_session_factory
    from app.ingestion.tasks import build_deps

    if get_settings().environment == "production":
        print("Refusing to seed a production environment.", file=sys.stderr)
        return 1

    with get_session_factory()() as session:
        seed_users(session, password)
        session.commit()
        identifiers = seed_documents(session, build_deps())
    if args.testset:
        write_testset(identifiers, args.testset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
