import os
import sys
from pathlib import Path

from dotenv import dotenv_values
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[1]
ENV_FILE = BACKEND_DIR / ".env"
APP_DATA_ROOT = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "StealthInterviewCoPilot"
DB_FILENAME = "local_copilot.db"

# Private-network browsers (phone on the same Wi-Fi) are allowed to call the API.
LAN_ORIGIN_REGEX = (
    r"^https?://("
    r"localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\]"
    r"|192\.168\.\d{1,3}\.\d{1,3}"
    r"|10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
    r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
    r")(?::\d+)?$"
    # The hosted web app and every Vercel preview deployment: the project domain
    # can change without a backend redeploy or a CORS failure on the phone.
    r"|^https://[a-z0-9-]+(?:\.[a-z0-9-]+)*\.vercel\.app$"
)


def _default_database_url() -> str:
    """Anchor SQLite to one file so every launcher (bat, desktop, exe) shares it."""
    if getattr(sys, "frozen", False):
        APP_DATA_ROOT.mkdir(parents=True, exist_ok=True)
        database_path = APP_DATA_ROOT / DB_FILENAME
    else:
        database_path = BACKEND_DIR / DB_FILENAME
    return f"sqlite:///{database_path.as_posix()}"


class Settings(BaseSettings):
    groq_api_key: str | None = None
    host: str = "0.0.0.0"
    port: int = 8000
    database_url: str = _default_database_url()
    cors_origins: str = "http://localhost:3000"
    cors_origin_regex: str = LAN_ORIGIN_REGEX
    pairing_ttl_seconds: int = 600
    answer_model: str = "llama-3.3-70b-versatile"
    resume_model: str = "openai/gpt-oss-120b"
    answer_temperature: float = 0.6

    model_config = SettingsConfigDict(env_file=str(ENV_FILE), env_file_encoding="utf-8", extra="ignore")

    @field_validator("port")
    @classmethod
    def _valid_port(cls, value: int) -> int:
        # A stray PORT=0 in the environment must not make the desktop app bind a random port.
        return value if 1 <= value <= 65535 else 8000

    @field_validator("host")
    @classmethod
    def _valid_host(cls, value: str) -> str:
        return value.strip() or "0.0.0.0"

    @field_validator("database_url")
    @classmethod
    def _anchor_sqlite_path(cls, value: str) -> str:
        """Turn sqlite:///./local_copilot.db into one absolute file shared by every launcher."""
        prefix = "sqlite:///"
        if not value.startswith(prefix):
            return value
        raw = value[len(prefix):].replace("\\", "/")
        if raw in {"", ":memory:"}:
            return value
        if raw.startswith("./"):
            raw = raw[2:]
        path = Path(raw)
        if path.is_absolute():
            return f"{prefix}{path.as_posix()}"
        base = APP_DATA_ROOT if getattr(sys, "frozen", False) else BACKEND_DIR
        return f"{prefix}{(base / path).as_posix()}"

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def is_reasoning_model(self) -> bool:
        return "gpt-oss" in self.answer_model.lower()

    @property
    def database_path(self) -> Path | None:
        prefix = "sqlite:///"
        if not self.database_url.startswith(prefix):
            return None
        return Path(self.database_url[len(prefix):])


settings = Settings()

_KEY_CACHE: dict[str, tuple[float, str]] = {}


def _key_from_file(path: Path, variable: str) -> str:
    """Read one API key, re-reading only when the file actually changes."""
    try:
        modified = path.stat().st_mtime
    except OSError:
        return ""
    cache_key = f"{path}:{variable}"
    cached = _KEY_CACHE.get(cache_key)
    if cached and cached[0] == modified:
        return cached[1]
    try:
        value = str(dotenv_values(path).get(variable) or "").strip()
    except (OSError, ValueError):
        value = ""
    _KEY_CACHE[cache_key] = (modified, value)
    return value


def _get_api_key(variable: str) -> str | None:
    """Resolve an API key at call time for source, portable, and installed builds."""
    environment_key = os.environ.get(variable, "").strip()
    if environment_key:
        return environment_key

    candidates = [ENV_FILE]
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        candidates.extend((
            executable_dir / ".env",
            executable_dir / "backend" / ".env",
            executable_dir.parent / "backend" / ".env",
            executable_dir.parent.parent / "backend" / ".env",
        ))
    candidates.append(APP_DATA_ROOT / ".env")

    for path in candidates:
        file_key = _key_from_file(path, variable)
        if file_key:
            return file_key
    return None


def get_groq_api_key() -> str | None:
    """Groq powers resume parsing, speech transcription, and interview answers."""
    return _get_api_key("GROQ_API_KEY")
