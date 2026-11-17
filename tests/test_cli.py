"""End-to-end CLI tests: real argparse, real graph, offline only."""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

from looprag.cli import main
from looprag.corpus import make_store, save_store

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "examples" / "corpus"
QUESTION = "How is the station powered in winter, and what happens during a power emergency?"


class NonInteractiveStdin:
    def isatty(self) -> bool:
        return False

    def readline(self) -> str:
        return ""


class EofPseudoTtyStdin:
    """Mimics Windows NUL redirection: isatty() is true but reads hit EOF."""

    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        return ""


class InterruptStdin:
    def isatty(self) -> bool:
        return True

    def readline(self) -> str:
        raise KeyboardInterrupt


def _ingest(tmp_path: Path) -> Path:
    store = tmp_path / "store.json"
    assert main(["ingest", str(CORPUS), "--store", str(store)]) == 0
    return store


def test_ingest_then_ask_json(tmp_path: Path, capsys) -> None:
    store = _ingest(tmp_path)
    capsys.readouterr()
    code = main(["ask", QUESTION, "--store", str(store), "--offline", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["final"]
    assert "[1]" in payload["final"]
    assert payload["grounded"] is True
    assert payload["insufficient"] is False
    assert payload["attempts"] == 1
    assert payload["mode"] == "offline-heuristic"
    assert payload["thread"]
    assert payload["query"]
    assert payload["sources"]
    for source in payload["sources"]:
        assert {"source", "title", "chunk", "id", "score"} <= set(source)
    nodes = [event["node"] for event in payload["trace"]]
    assert nodes[0] == "retrieve"
    assert nodes[-1] == "finalize"
    assert "review_round" in payload


def test_ask_auto_ingests_examples_corpus(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    shutil.copytree(CORPUS, tmp_path / "examples" / "corpus")
    code = main(["ask", QUESTION, "--offline", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["final"]
    assert (tmp_path / ".looprag" / "corpus.json").exists()
    assert "ingested sample corpus" in captured.err


def test_missing_custom_store_returns_2(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "ask",
            QUESTION,
            "--store",
            str(tmp_path / "missing.json"),
            "--offline",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "ingest" in captured.err


def test_review_with_non_interactive_stdin_exits_3(tmp_path: Path, monkeypatch, capsys) -> None:
    store = _ingest(tmp_path)
    capsys.readouterr()
    monkeypatch.setattr(sys, "stdin", NonInteractiveStdin())
    code = main(
        [
            "ask",
            QUESTION,
            "--store",
            str(store),
            "--offline",
            "--review",
            "--json",
        ]
    )
    captured = capsys.readouterr()
    assert code == 3
    assert captured.out == ""
    assert "not interactive" in captured.err


def test_chat_with_non_interactive_stdin_returns_2(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "stdin", NonInteractiveStdin())
    code = main(["chat", "--offline"])
    captured = capsys.readouterr()
    assert code == 2
    assert "interactive" in captured.err


def test_chat_with_eof_pseudo_tty_returns_2(tmp_path: Path, monkeypatch, capsys) -> None:
    store = _ingest(tmp_path)
    capsys.readouterr()
    monkeypatch.setattr(sys, "stdin", EofPseudoTtyStdin())
    code = main(["chat", "--offline", "--store", str(store)])
    captured = capsys.readouterr()
    assert code == 2
    assert "interactive" in captured.err


def test_chat_ctrl_c_exits_cleanly(tmp_path: Path, monkeypatch, capsys) -> None:
    store = _ingest(tmp_path)
    capsys.readouterr()
    monkeypatch.setattr(sys, "stdin", InterruptStdin())
    code = main(["chat", "--offline", "--store", str(store)])
    assert code == 0


def test_review_exit_3_reports_resumable_thread(tmp_path: Path, monkeypatch, capsys) -> None:
    store = _ingest(tmp_path)
    capsys.readouterr()
    monkeypatch.setattr(sys, "stdin", NonInteractiveStdin())
    database = tmp_path / "review.sqlite"
    code = main(
        [
            "ask",
            QUESTION,
            "--store",
            str(store),
            "--offline",
            "--review",
            "--json",
            "--sqlite",
            str(database),
        ]
    )
    captured = capsys.readouterr()
    assert code == 3
    assert captured.out == ""
    match = re.search(r"review checkpoint: thread ([0-9a-f-]{36})", captured.err)
    assert match, captured.err
    thread = match.group(1)
    assert f'"thread": "{thread}"' in captured.err
    assert database.exists()


def test_ask_with_tokenless_store_returns_insufficient(tmp_path: Path, capsys) -> None:
    store = tmp_path / "chinese.json"
    chunks = [
        {
            "id": "a" * 16,
            "source": "zh.md",
            "title": "站",
            "chunk": 0,
            "content": "站的中文内容",
        }
    ]
    save_store(make_store(chunks, 900, 120), store)
    code = main(["ask", "这个站是什么?", "--store", str(store), "--offline", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["insufficient"] is True
    assert "enough evidence" in payload["final"].lower()


def test_directory_store_is_a_config_error(tmp_path: Path, capsys) -> None:
    dirstore = tmp_path / "dirstore"
    dirstore.mkdir()
    code = main(["ask", QUESTION, "--store", str(dirstore), "--offline", "--json"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "store" in captured.err.lower()


def test_ingest_to_directory_store_is_a_config_error(tmp_path: Path, capsys) -> None:
    dirstore = tmp_path / "dirstore"
    dirstore.mkdir()
    code = main(["ingest", str(CORPUS), "--store", str(dirstore)])
    captured = capsys.readouterr()
    assert code == 2
    assert "store" in captured.err.lower()


def test_sqlite_path_that_is_a_directory_returns_2(tmp_path: Path, capsys) -> None:
    store = _ingest(tmp_path)
    capsys.readouterr()
    dirdb = tmp_path / "dbdir"
    dirdb.mkdir()
    code = main(
        [
            "ask",
            QUESTION,
            "--store",
            str(store),
            "--offline",
            "--json",
            "--sqlite",
            str(dirdb),
        ]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "SQLite" in captured.err


def test_version_flag(capsys) -> None:
    assert main(["--version"]) == 0
    assert "looprag 0.1.0" in capsys.readouterr().out


def test_invalid_chunk_size_returns_2(tmp_path: Path, capsys) -> None:
    code = main(
        [
            "ingest",
            str(CORPUS),
            "--store",
            str(tmp_path / "bad.json"),
            "--chunk-size",
            "0",
        ]
    )
    assert code == 2
    assert "chunk-size" in capsys.readouterr().err


def test_unknown_command_is_usage_error(capsys) -> None:
    code = main(["frobnicate"])
    assert code == 2
    assert capsys.readouterr().err


def test_ask_with_sqlite_checkpoints_round_trips(tmp_path: Path, capsys) -> None:
    store = _ingest(tmp_path)
    capsys.readouterr()
    database = tmp_path / "checkpoints" / "state.sqlite"
    code = main(
        [
            "ask",
            QUESTION,
            "--store",
            str(store),
            "--offline",
            "--json",
            "--sqlite",
            str(database),
            "--thread",
            "demo-thread",
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)
    assert payload["thread"] == "demo-thread"
    assert database.exists()
