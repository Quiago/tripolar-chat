"""Database engine, session factory, and startup checks.

SQLite notes:
- Paths are always resolved to absolute so the DB location is invariant
  regardless of the working directory when uvicorn is launched.
- WAL journal mode is enabled automatically; it prevents the "database is
  locked" errors that appear when uvicorn hot-reloads alongside an open
  write transaction.
"""

import logging
import os
from pathlib import Path

from sqlalchemy import event, text
from sqlalchemy.engine import Engine
from sqlmodel import SQLModel, Session, create_engine

from server.config import settings

log = logging.getLogger(__name__)

# ── Project root = directory that contains this file's parent (server/) ───────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Path helpers ───────────────────────────────────────────────────────────────

def _resolve_url(url: str) -> str:
    """Return *url* with any relative SQLite path expanded to absolute.

    Anchors relative paths at the project root so the DB location is the
    same whether uvicorn is started from the project root, the server/
    sub-directory, or anywhere else.
    """
    if not url.startswith("sqlite"):
        return url

    # sqlite+driver:///PATH  →  split after the third slash
    prefix, _, path_str = url.partition("///")
    p = Path(path_str)
    if not p.is_absolute():
        p = (_PROJECT_ROOT / p).resolve()

    return f"{prefix}///{p}"


def _ensure_dir(url: str) -> None:
    """Create the parent directory for the SQLite file if it does not exist."""
    if not url.startswith("sqlite"):
        return
    path_str = url.split("///", 1)[-1]
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def _check_writable(url: str) -> None:
    """Raise RuntimeError with a human-readable fix if the DB is not writable.

    Called once at startup so that permission errors surface immediately with
    a useful message instead of exploding on the first write request.
    """
    if not url.startswith("sqlite"):
        return

    db_path = Path(url.split("///", 1)[-1])

    if db_path.exists():
        if not os.access(db_path, os.W_OK):
            raise RuntimeError(
                f"SQLite database is read-only: {db_path}\n"
                f"  Fix: chmod 664 '{db_path}'"
            )
    else:
        parent = db_path.parent
        if parent.exists() and not os.access(parent, os.W_OK):
            raise RuntimeError(
                f"Database directory is not writable: {parent}\n"
                f"  Fix: chmod 775 '{parent}'"
            )


# ── Engine setup ───────────────────────────────────────────────────────────────

_resolved_url = _resolve_url(settings.database_url)
_ensure_dir(_resolved_url)
_check_writable(_resolved_url)

log.info("Database: %s", _resolved_url)

connect_args = (
    {"check_same_thread": False} if _resolved_url.startswith("sqlite") else {}
)
engine = create_engine(_resolved_url, connect_args=connect_args, echo=False)


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_conn, _connection_record) -> None:
    """Enable WAL mode and foreign-key enforcement for every new SQLite connection."""
    if "sqlite" not in _resolved_url:
        return
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


# ── Public API ─────────────────────────────────────────────────────────────────

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_db():
    with Session(engine) as session:
        yield session
