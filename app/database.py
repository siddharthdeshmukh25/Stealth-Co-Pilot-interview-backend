import shutil
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from .config import BACKEND_DIR, settings


def _adopt_legacy_database() -> None:
    """Move a database created by an older relative-path build into the new location."""
    target = settings.database_path
    if target is None or target.exists():
        return
    candidates = (Path.cwd() / "local_copilot.db", BACKEND_DIR / "local_copilot.db")
    for candidate in candidates:
        try:
            if not candidate.is_file() or candidate.resolve() == target.resolve():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, target)
            return
        except OSError:
            continue


_adopt_legacy_database()

connect_args = {"check_same_thread": False, "timeout": 15} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args, echo=False)


def _add_missing_columns() -> None:
    """create_all never alters an existing table: add the history columns once."""
    if not settings.database_url.startswith("sqlite"):
        return
    additions = {
        "ended_at": "ALTER TABLE interview_sessions ADD COLUMN ended_at DATETIME",
        "duration_seconds": (
            "ALTER TABLE interview_sessions ADD COLUMN duration_seconds INTEGER NOT NULL DEFAULT 0"
        ),
    }
    with engine.connect() as connection:
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(interview_sessions)")}
        if not columns:
            return
        for name, statement in additions.items():
            if name not in columns:
                connection.exec_driver_sql(statement)
        connection.commit()


def init_db() -> None:
    from . import models  # noqa: F401
    SQLModel.metadata.create_all(engine)
    _add_missing_columns()


def get_session():
    with Session(engine) as session:
        yield session
