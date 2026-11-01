"""Test setup: a real PostgreSQL 16, migrated with Alembic.

The environment is set before anything from `app` is imported, because the settings
object and the Celery app read it at import time.
"""

import os
import shutil
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path

import pgserver

_temp_root = Path(tempfile.mkdtemp(prefix="campusmind-tests-"))
_postgres = pgserver.get_server(_temp_root / "pg", cleanup_mode="stop")

os.environ.update(
    {
        "ENVIRONMENT": "test",
        "DATABASE_URL": _postgres.get_uri().replace("postgresql://", "postgresql+psycopg://", 1),
        "REDIS_URL": "redis://localhost:6399/15",
        "CELERY_EAGER": "true",
        "UPLOAD_DIR": str(_temp_root / "uploads"),
        "LOG_LEVEL": "WARNING",
        "SECRET_KEY": "test-secret-key-test-secret-key-0123456789",
    }
)

import pytest  # noqa: E402
from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402

from app.auth.tokens import create_access_token  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.db import get_engine, get_session_factory  # noqa: E402
from app.ingestion.pipeline import PipelineDeps  # noqa: E402
from app.ingestion.storage import LocalStorage  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import Base, User  # noqa: E402
from tests.helpers import FakeIndexer, FakeLLM, FakeOcr  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
BACKEND_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session", autouse=True)
def _database() -> Iterator[None]:
    """Run the real migrations once, and stop Postgres at the end."""
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    command.upgrade(config, "head")
    yield
    get_engine().dispose()
    _postgres.cleanup()
    shutil.rmtree(_temp_root, ignore_errors=True)


@pytest.fixture
def db() -> Iterator[Session]:
    """A session, with every table emptied afterwards."""
    session = get_session_factory()()
    yield session
    session.rollback()
    session.close()
    tables = ", ".join(f'"{table.name}"' for table in Base.metadata.sorted_tables)
    with get_engine().begin() as connection:
        connection.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))


@pytest.fixture
def storage(tmp_path: Path) -> LocalStorage:
    """File storage in a temporary folder."""
    return LocalStorage(tmp_path / "uploads")


@pytest.fixture
def deps(db: Session, storage: LocalStorage) -> PipelineDeps:
    """Pipeline dependencies with fake OCR, LLM and indexer."""
    return PipelineDeps(
        session_factory=get_session_factory(),
        storage=storage,
        ocr=FakeOcr("The library will remain closed on Friday for annual stock verification."),
        llm=FakeLLM(),
        indexer=FakeIndexer(),
        settings=get_settings(),
    )


@pytest.fixture
def client() -> TestClient:
    """An API client."""
    return TestClient(create_app())


@pytest.fixture
def make_user(db: Session) -> Callable[..., User]:
    """Create users in the database."""
    counter = iter(range(1, 10_000))

    def factory(role: str = "student", **overrides: object) -> User:
        number = next(counter)
        fields = {
            "email": f"{role}{number}@college.edu",
            "full_name": f"Test {role.title()} {number}",
            "role": role,
            **overrides,
        }
        user = User(**fields)
        db.add(user)
        db.commit()
        return user

    return factory


def auth_header(user: User) -> dict[str, str]:
    """Return an Authorization header for a user."""
    return {"Authorization": f"Bearer {create_access_token(user.id, user.role)}"}
