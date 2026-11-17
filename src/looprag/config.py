"""Environment and provider settings for LoopRAG.

A deliberately tiny ``.env`` loader: no python-dotenv dependency, only the
``LOOPRAG_*`` keys, never overriding variables already present in the
environment. Values are never logged.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_KEYS = ("LOOPRAG_API_KEY", "LOOPRAG_BASE_URL", "LOOPRAG_MODEL")
DEFAULT_MODEL = "gpt-4o-mini"


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Load only ``LOOPRAG_*`` keys from ``path`` without overriding env vars."""
    loaded: dict[str, str] = {}
    target = Path(path)
    if not target.is_file():
        return loaded
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return loaded
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in ENV_KEYS or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value
        loaded[key] = value
    return loaded


@dataclass(frozen=True)
class Settings:
    model: str
    api_key: str | None
    base_url: str | None

    @property
    def online(self) -> bool:
        return bool(self.api_key)

    @property
    def mode(self) -> str:
        return f"llm:{self.model}" if self.online else "offline-heuristic"


def resolve_settings(
    model: str | None = None,
    base_url: str | None = None,
    env: dict[str, str] | None = None,
) -> Settings:
    source = os.environ if env is None else env
    api_key = source.get("LOOPRAG_API_KEY") or source.get("OPENAI_API_KEY") or None
    resolved_model = model or source.get("LOOPRAG_MODEL") or DEFAULT_MODEL
    resolved_base = base_url or source.get("LOOPRAG_BASE_URL") or None
    return Settings(model=resolved_model, api_key=api_key, base_url=resolved_base)
