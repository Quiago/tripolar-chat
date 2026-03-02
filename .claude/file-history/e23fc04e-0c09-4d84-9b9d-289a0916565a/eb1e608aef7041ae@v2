from pathlib import Path

from sqlmodel import SQLModel, create_engine, Session

from server.config import settings


def _prepare_sqlite(url: str) -> None:
    """Create parent directory for SQLite databases if needed."""
    if url.startswith("sqlite"):
        path_part = url.split("///")[-1]
        Path(path_part).parent.mkdir(parents=True, exist_ok=True)


_prepare_sqlite(settings.database_url)

connect_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)
engine = create_engine(settings.database_url, connect_args=connect_args, echo=False)


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_db():
    with Session(engine) as session:
        yield session
