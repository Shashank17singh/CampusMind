"""Run a local PostgreSQL 16 without Docker, for development.

Usage: python scripts/devdb.py [command...]

With a command, runs it with DATABASE_URL set and then stops the server. Without one,
prints the URL and keeps the server running until interrupted.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pgserver

DATA_DIR = Path(__file__).resolve().parent.parent / ".pgdata"


def main() -> int:
    """Start the server and run the requested command."""
    server = pgserver.get_server(DATA_DIR, cleanup_mode="stop")
    url = server.get_uri().replace("postgresql://", "postgresql+psycopg://", 1)

    if len(sys.argv) > 1:
        environment = {**os.environ, "DATABASE_URL": url}
        return subprocess.run(sys.argv[1:], env=environment, check=False).returncode  # noqa: S603

    print(f"DATABASE_URL={url}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
