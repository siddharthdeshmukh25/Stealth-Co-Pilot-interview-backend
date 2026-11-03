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


def init_db() -> None:
    from . import models  # noqa: F401
    SQLModel.metadata.create_all(engine)


def get_session():
    with Session(engine) as session:
        yield session
