"""Privacy and hygiene checks over every shipped file."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    ".looprag",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
    "build",
    "dist",
}
TEXT_SUFFIXES = {
    ".md",
    ".py",
    ".toml",
    ".lock",
    ".yml",
    ".yaml",
    ".txt",
    ".example",
    ".gitignore",
    "",
}


def _shipped_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in EXCLUDED_DIRS for part in path.parts):
            continue
        if path.suffix not in TEXT_SUFFIXES and path.name != ".gitignore":
            continue
        files.append(path)
    return files


def test_required_files_exist() -> None:
    for name in (
        "README.md",
        "README.zh-CN.md",
        "LICENSE",
        "pyproject.toml",
        "uv.lock",
        ".gitignore",
        ".env.example",
        ".github/workflows/ci.yml",
    ):
        assert (REPO_ROOT / name).is_file(), f"missing {name}"


def test_no_env_file_and_gitignore_covers_runtime_data() -> None:
    assert not (REPO_ROOT / ".env").exists()
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    for entry in (".env", ".venv", ".looprag", "*.sqlite"):
        assert entry in ignore


def _forbidden_patterns() -> list[re.Pattern[str]]:
    windows_home = "C:" + "\\\\Users\\\\"
    windows_home_forward = "C:" + "/Users/"
    return [
        re.compile(re.escape(windows_home), re.IGNORECASE),
        re.compile(re.escape(windows_home_forward), re.IGNORECASE),
        # Detect any personal QQ address without embedding a real one here.
        re.compile(r"\b\d{5,12}@qq\.com\b", re.IGNORECASE),
        re.compile(r"\bghp_[A-Za-z0-9]{20,}"),
        re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
        re.compile(r"\bsk-[A-Za-z0-9]{20,}"),
        re.compile(r"AKIA[0-9A-Z]{16}"),
        re.compile("Generated with " + "Claude", re.IGNORECASE),
        re.compile("Co-Authored-By: " + "Claude", re.IGNORECASE),
    ]


def test_shipped_files_have_no_private_or_secret_material() -> None:
    patterns = _forbidden_patterns()
    hits: list[str] = []
    for path in _shipped_files():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        relative = path.relative_to(REPO_ROOT).as_posix()
        for pattern in patterns:
            if pattern.search(text):
                hits.append(f"{relative}: {pattern.pattern}")
    assert not hits, "forbidden content found: " + "; ".join(hits)


def test_readmes_have_language_switch_links() -> None:
    english = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (REPO_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    assert "README.zh-CN.md" in english
    assert "README.md" in chinese
    assert "github.com/jmlistening/langgraph-looprag" in english
    assert "github.com/jmlistening/langgraph-looprag" in chinese


def test_no_runtime_store_tracked_in_manifest() -> None:
    # Nothing here runs git; instead assert the ignore file covers generated data
    # and that no committed-looking runtime files sit at the repo root.
    ignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "__pycache__" in ignore
    assert ".pytest_cache" in ignore
    assert ".ruff_cache" in ignore
    root_sqlite = list(REPO_ROOT.glob("*.sqlite")) + list(REPO_ROOT.glob("*.db"))
    assert root_sqlite == []
