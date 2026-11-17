"""LoopRAG command line interface."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

from . import __version__
from .config import load_dotenv, resolve_settings
from .corpus import (
    DEFAULT_STORE,
    BM25Retriever,
    CorpusError,
    load_chunks,
    load_store,
    make_store,
    save_store,
)
from .graph import build_graph, run_config
from .heuristic import HeuristicPolicy
from .llm_policy import LLMPolicy
from .output import (
    make_console,
    print_update,
    render_ingest,
    render_result,
    render_review,
    render_sources,
    render_trace,
)
from .policy import PolicyError
from .state import Chunk

DEFAULT_CORPUS = Path("examples/corpus")
DEFAULT_SQLITE = Path(".looprag/checkpoints.sqlite")
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 120
RECURSION_LIMIT = 60


class UsageError(RuntimeError):
    """Raised for command-line combinations the CLI refuses."""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="looprag",
        description=(
            "A self-correcting RAG agent loop: retrieve, grade, generate, verify, "
            "rewrite, and optionally review by a human."
        ),
    )
    parser.add_argument("--version", action="version", version=f"looprag {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="chunk documents and build a corpus store")
    ingest.add_argument("source", help="file or directory (.md, .markdown, .txt, .pdf)")
    ingest.add_argument("--store", default=str(DEFAULT_STORE), help="store path")
    ingest.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    ingest.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--store", default=str(DEFAULT_STORE))
    ask.add_argument(
        "--offline", action="store_true", help="force the deterministic heuristic policy"
    )
    ask.add_argument("--k", type=int, default=4, help="chunks to retrieve")
    ask.add_argument("--max-attempts", type=int, default=2, help="retrieval/verification attempts")
    ask.add_argument("--thread", default=None, help="checkpoint thread id (UUID4 when absent)")
    ask.add_argument(
        "--sqlite",
        nargs="?",
        const=str(DEFAULT_SQLITE),
        default=None,
        help="SQLite checkpoint path (default .looprag/checkpoints.sqlite with sqlite extra)",
    )
    ask.add_argument("--review", action="store_true", help="human-in-the-loop review")
    ask.add_argument("--show-docs", action="store_true", help="print retrieved chunks")
    ask.add_argument("--json", dest="json_output", action="store_true", help="emit one JSON object")
    ask.add_argument("--model", default=None)
    ask.add_argument("--base-url", default=None)

    chat = sub.add_parser("chat", help="interactive multi-turn loop")
    chat.add_argument("--store", default=str(DEFAULT_STORE))
    chat.add_argument("--offline", action="store_true")
    chat.add_argument("--k", type=int, default=4)
    chat.add_argument("--max-attempts", type=int, default=2)
    chat.add_argument("--thread", default=None)
    chat.add_argument(
        "--sqlite",
        nargs="?",
        const=str(DEFAULT_SQLITE),
        default=None,
        help="SQLite checkpoint path (default .looprag/checkpoints.sqlite with sqlite extra)",
    )
    chat.add_argument("--review", action="store_true")
    chat.add_argument("--model", default=None)
    chat.add_argument("--base-url", default=None)
    return parser


def _validate(args: argparse.Namespace) -> None:
    if hasattr(args, "chunk_size"):
        if args.chunk_size <= 0:
            raise UsageError("--chunk-size must be positive")
        if args.chunk_overlap < 0 or args.chunk_overlap >= args.chunk_size:
            raise UsageError("--chunk-overlap must be >= 0 and smaller than --chunk-size")
    if hasattr(args, "k") and args.k <= 0:
        raise UsageError("--k must be positive")
    if hasattr(args, "max_attempts") and args.max_attempts <= 0:
        raise UsageError("--max-attempts must be positive")


def _stdin_is_interactive() -> bool:
    try:
        return bool(sys.stdin and sys.stdin.isatty())
    except (AttributeError, ValueError):
        return False


def _ensure_store(path: Path) -> None:
    if path.exists():
        return
    if path == DEFAULT_STORE:
        if not DEFAULT_CORPUS.exists():
            raise CorpusError(
                f"Default corpus store not found and {DEFAULT_CORPUS} is unavailable. "
                "Run: looprag ingest <SOURCE>"
            )
        chunks = load_chunks(DEFAULT_CORPUS)
        save_store(make_store(chunks, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP), path)
        print(
            "notice: ingested sample corpus from "
            f"{DEFAULT_CORPUS.as_posix()} into {path.as_posix()}",
            file=sys.stderr,
        )
        return
    raise CorpusError(
        f"Corpus store not found: {path}. "
        f"Build it first with: looprag ingest <SOURCE> --store {path}"
    )


def _build_policy(args: argparse.Namespace) -> tuple[HeuristicPolicy | LLMPolicy, bool]:
    if args.offline:
        return HeuristicPolicy(), False
    load_dotenv()
    settings = resolve_settings(model=args.model, base_url=args.base_url)
    if not settings.online:
        print(
            "notice: no API key found (LOOPRAG_API_KEY / OPENAI_API_KEY); "
            "using offline heuristic mode.",
            file=sys.stderr,
        )
        return HeuristicPolicy(), False
    return LLMPolicy(
        model=settings.model, api_key=settings.api_key or "", base_url=settings.base_url
    ), True


@contextmanager
def _open_checkpointer(args: argparse.Namespace):
    if args.sqlite:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError as exc:
            raise UsageError(
                "SQLite checkpoints require the optional sqlite dependency. "
                "Install it with: uv sync --extra sqlite (or pip install 'looprag[sqlite]')"
            ) from exc
        path = Path(args.sqlite)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(str(path), check_same_thread=False)
        except (OSError, sqlite3.Error) as exc:
            raise UsageError(f"Could not open SQLite checkpoint database: {args.sqlite}") from exc
        try:
            yield SqliteSaver(connection)
        finally:
            connection.close()
        return
    from langgraph.checkpoint.memory import InMemorySaver

    yield InMemorySaver()


def _initial_state(question: str) -> dict:
    return {
        "question": question,
        "query": question,
        "attempt": 0,
        "documents": [],
        "grades": [],
        "relevant": [],
        "answer": "",
        "critique": None,
        "grounded": False,
        "final": "",
        "insufficient": False,
        "review_round": 0,
        "approved": None,
        "notice": None,
        "trace": [],
    }


def _prompt(message: str, json_mode: bool = False) -> str:
    stream = sys.stderr if json_mode else sys.stdout
    print(message, end="", file=stream, flush=True)
    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    return line.rstrip("\r\n")


def _interrupt_payload(interrupts) -> dict:
    first = interrupts[0] if isinstance(interrupts, (list, tuple)) and interrupts else interrupts
    value = getattr(first, "value", first)
    return value if isinstance(value, dict) else {"draft": str(value)}


def _run_with_review(graph, initial: dict, config: dict, console, json_mode: bool, *, thread: str):
    from langgraph.types import Command

    current = graph.invoke(initial, config)
    while True:
        interrupts = current.get("__interrupt__") if isinstance(current, dict) else None
        if not interrupts:
            return dict(current)
        payload = _interrupt_payload(interrupts)
        if not _stdin_is_interactive():
            if json_mode:
                print(
                    json.dumps(
                        {"status": "interrupted", "thread": thread, "review": payload},
                        ensure_ascii=False,
                    ),
                    file=sys.stderr,
                )
            else:
                render_review(console, payload)
            print(
                "Review is required but stdin is not interactive; exiting with status 3.",
                file=sys.stderr,
            )
            return None
        if json_mode:
            print(
                "review required: draft in stderr context; "
                + json.dumps(payload, ensure_ascii=False),
                file=sys.stderr,
            )
        else:
            render_review(console, payload)
        try:
            answer = _prompt("Approve? [y/N] ", json_mode)
            approved = answer.strip().lower() in {"y", "yes"}
            feedback = ""
            if not approved:
                feedback = _prompt("Feedback (optional): ", json_mode)
        except EOFError:
            print(
                "Review is required but stdin reached EOF; exiting with status 3.",
                file=sys.stderr,
            )
            return None
        current = graph.invoke(Command(resume={"approved": approved, "feedback": feedback}), config)


def _sources_payload(state: dict) -> list[dict]:
    docs: list[Chunk] = list(state.get("relevant") or state.get("documents", []))
    return [
        {
            "source": doc.get("source"),
            "title": doc.get("title"),
            "chunk": doc.get("chunk"),
            "id": doc.get("id"),
            "score": float(doc.get("score", 0.0)),
        }
        for doc in docs
    ]


def _json_payload(state: dict, *, question: str, mode: str, thread: str) -> dict:
    return {
        "question": state.get("question") or question,
        "final": state.get("final") or "",
        "grounded": bool(state.get("grounded", False)),
        "insufficient": bool(state.get("insufficient", False)),
        "attempts": int(state.get("attempt", 0)),
        "query": state.get("query") or question,
        "mode": mode,
        "thread": thread,
        "trace": state.get("trace", []),
        "sources": _sources_payload(state),
        "review_round": int(state.get("review_round", 0)),
    }


def _cmd_ingest(args: argparse.Namespace, console) -> int:
    chunks = load_chunks(args.source, args.chunk_size, args.chunk_overlap)
    if not chunks:
        raise CorpusError("No text could be extracted from the source.")
    store = make_store(chunks, args.chunk_size, args.chunk_overlap)
    save_store(store, args.store)
    files = len({chunk.get("source", "") for chunk in chunks})
    render_ingest(console, files=files, chunks=len(chunks), store=str(args.store))
    return 0


def _cmd_ask(args: argparse.Namespace, console) -> int:
    store_path = Path(args.store)
    _ensure_store(store_path)
    chunks = load_store(store_path)
    if not chunks:
        raise CorpusError(f"Corpus store is empty: {store_path}")
    retriever = BM25Retriever(chunks, default_k=args.k)
    policy, _online = _build_policy(args)
    thread = args.thread or str(uuid.uuid4())
    config = run_config(thread, RECURSION_LIMIT)
    initial = _initial_state(args.question)
    if args.review:
        checkpoint = Path(args.sqlite).as_posix() if args.sqlite else "in-memory"
        print(
            f"review checkpoint: thread {thread}, checkpointer {checkpoint}",
            file=sys.stderr,
        )
    with _open_checkpointer(args) as checkpointer:
        graph = build_graph(
            policy,
            retriever,
            max_attempts=args.max_attempts,
            enable_review=args.review,
            checkpointer=checkpointer,
        )
        if args.review:
            state = _run_with_review(
                graph, initial, config, console, args.json_output, thread=thread
            )
            if state is None:
                return 3
        elif args.json_output:
            state = dict(graph.invoke(initial, config))
        else:
            for update in graph.stream(initial, config, stream_mode="updates"):
                print_update(console, update)
            state = dict(graph.get_state(config).values)
    if args.json_output:
        payload = _json_payload(state, question=args.question, mode=policy.name, thread=thread)
        print(json.dumps(payload, ensure_ascii=False))
    else:
        render_result(console, state, mode=policy.name, thread=thread, show_docs=args.show_docs)
    return 0


def _cmd_chat(args: argparse.Namespace, console) -> int:
    if not _stdin_is_interactive():
        print("error: chat requires an interactive stdin", file=sys.stderr)
        return 2
    store_path = Path(args.store)
    _ensure_store(store_path)
    chunks = load_store(store_path)
    if not chunks:
        raise CorpusError(f"Corpus store is empty: {store_path}")
    retriever = BM25Retriever(chunks, default_k=args.k)
    policy, _online = _build_policy(args)
    thread = args.thread or str(uuid.uuid4())
    config = run_config(thread, RECURSION_LIMIT)
    with _open_checkpointer(args) as checkpointer:
        graph = build_graph(
            policy,
            retriever,
            max_attempts=args.max_attempts,
            enable_review=args.review,
            checkpointer=checkpointer,
        )
        console.print(
            f"[bold]LoopRAG chat[/bold] - mode [cyan]{policy.name}[/cyan], "
            f"thread [dim]{thread}[/dim]. /help for commands."
        )
        last: dict = {}
        interactive_seen = False
        while True:
            try:
                line = _prompt("looprag> ")
            except KeyboardInterrupt:
                console.print("")
                return 0
            except EOFError:
                console.print("")
                if not interactive_seen:
                    # Some Windows non-console stdin handles (for example NUL)
                    # report isatty() == True but reach EOF immediately.
                    print("error: chat requires an interactive stdin", file=sys.stderr)
                    return 2
                return 0
            text = line.strip()
            if not text:
                continue
            interactive_seen = True
            if text in {"/exit", "/quit"}:
                return 0
            if text == "/help":
                console.print("/exit  /trace  /sources  /help")
                continue
            if text == "/trace":
                events = last.get("trace") or graph.get_state(config).values.get("trace", [])
                render_trace(console, list(events))
                continue
            if text == "/sources":
                render_sources(console, list(last.get("relevant") or last.get("documents", [])))
                continue
            try:
                initial = _initial_state(text)
                if args.review:
                    state = _run_with_review(
                        graph, initial, config, console, json_mode=False, thread=thread
                    )
                    if state is None:
                        return 0
                else:
                    for update in graph.stream(initial, config, stream_mode="updates"):
                        print_update(console, update)
                    state = dict(graph.get_state(config).values)
            except KeyboardInterrupt:
                console.print("")
                return 0
            except PolicyError as exc:
                print(f"error: {exc}", file=sys.stderr)
                continue
            last = state
            render_result(console, state, mode=policy.name, thread=thread)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 2
        return code
    console = make_console()
    try:
        _validate(args)
        if args.command == "ingest":
            return _cmd_ingest(args, console)
        if args.command == "ask":
            return _cmd_ask(args, console)
        if args.command == "chat":
            return _cmd_chat(args, console)
        raise UsageError(f"unknown command: {args.command}")
    except (UsageError, CorpusError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except PolicyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:  # pragma: no cover - piping edge case
        return 1
    except Exception as exc:  # noqa: BLE001 - a CLI must not dump a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1
