"""Rich rendering for human-facing CLI output. JSON mode never touches this."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .state import Chunk, TraceEvent


def make_console() -> Console:
    return Console(highlight=False)


def print_update(console: Console, update: dict) -> None:
    """Print the trace events carried by one ``stream_mode='updates'`` frame."""
    for node, payload in update.items():
        if node.startswith("__"):
            continue
        events: list[TraceEvent] = []
        if isinstance(payload, dict):
            events = payload.get("trace") or []
        if not events:
            console.print(Text.assemble((f"{node:<10}", "cyan")))
            continue
        for event in events:
            console.print(
                Text.assemble(
                    (f"{event.get('node', node):<10}", "cyan"),
                    str(event.get("message", "")),
                )
            )


def render_trace(console: Console, events: list[TraceEvent]) -> None:
    table = Table(title="Graph trace", show_lines=False, header_style="bold")
    table.add_column("node", style="cyan", no_wrap=True)
    table.add_column("message")
    for event in events:
        table.add_row(str(event.get("node", "")), str(event.get("message", "")))
    console.print(table)


def render_sources(console: Console, docs: list[Chunk]) -> None:
    if not docs:
        console.print("[dim]No sources retrieved.[/dim]")
        return
    table = Table(title="Sources", header_style="bold")
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("source", no_wrap=True)
    table.add_column("title")
    table.add_column("chunk", justify="right")
    table.add_column("score", justify="right")
    for number, doc in enumerate(docs, start=1):
        table.add_row(
            str(number),
            str(doc.get("source", "?")),
            str(doc.get("title", "")),
            str(doc.get("chunk", 0)),
            f"{float(doc.get('score', 0.0)):.3f}",
        )
    console.print(table)


def render_documents(console: Console, docs: list[Chunk]) -> None:
    if not docs:
        console.print("[dim]No documents retrieved.[/dim]")
        return
    console.print("[bold]Retrieved chunks[/bold]")
    for number, doc in enumerate(docs, start=1):
        snippet = " ".join(str(doc.get("content", "")).split())[:280]
        console.print(
            Text.assemble(
                (f"[{number}] ", "cyan"),
                (f"{doc.get('source', '?')} ", "bold"),
                (f"(score {float(doc.get('score', 0.0)):.3f}) ", "dim"),
                snippet + ("..." if len(str(doc.get("content", ""))) > 280 else ""),
            )
        )


def render_result(
    console: Console,
    state: dict,
    *,
    mode: str,
    thread: str,
    show_docs: bool = False,
) -> None:
    answer = str(state.get("final") or state.get("answer") or "")
    grounded = bool(state.get("grounded"))
    style = "green" if grounded else "yellow"
    console.print(Panel(answer or "(no answer)", title="Answer", border_style=style))
    docs = list(state.get("relevant") or state.get("documents", []))
    render_sources(console, docs)
    if show_docs:
        render_documents(console, docs)
    console.print(
        Text.assemble(
            ("mode ", "dim"),
            (mode, "bold"),
            ("   attempts ", "dim"),
            (str(int(state.get("attempt", 0))), "bold"),
            ("   thread ", "dim"),
            (thread, ""),
        )
    )
    notice = state.get("notice")
    if notice:
        console.print(Text.assemble(("warning: ", "yellow bold"), str(notice)))
    if not grounded and not state.get("insufficient"):
        console.print("[yellow]warning: the answer was not verified against sources.[/yellow]")


def render_review(console: Console, payload: dict) -> None:
    console.print(
        Panel(
            str(payload.get("draft", "")),
            title=f"Draft for review (round {payload.get('review_round', 1)})",
            border_style="magenta",
        )
    )
    console.print(f"[bold]Question:[/bold] {payload.get('question', '')}")
    console.print(
        f"[dim]attempt {payload.get('attempt', 0)} | "
        f"grounded={payload.get('grounded', False)}[/dim]"
    )
    for source in payload.get("sources", []):
        console.print(
            Text.assemble(
                (f"[{source.get('n', '?')}] ", "cyan"),
                (f"{source.get('source', '?')}: ", "bold"),
                str(source.get("snippet", "")),
            )
        )


def render_ingest(console: Console, *, files: int, chunks: int, store: str) -> None:
    table = Table(title="Ingest summary", header_style="bold")
    table.add_column("metric", style="cyan")
    table.add_column("value", justify="right")
    table.add_row("documents", str(files))
    table.add_row("chunks", str(chunks))
    table.add_row("store", store)
    console.print(table)
    console.print('[dim]Next: looprag ask "How is the station powered in winter?" --offline[/dim]')
