"""Corpus loading, paragraph-aware chunking, JSON store and BM25 retrieval."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from rank_bm25 import BM25Okapi

from .policy import STOPWORDS, stem, tokenize
from .state import Chunk

TEXT_SUFFIXES = {".md", ".markdown", ".txt"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | {".pdf"}
STORE_VERSION = 1
DEFAULT_STORE = Path(".looprag/corpus.json")


class CorpusError(RuntimeError):
    """Raised for unreadable sources or invalid stores."""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise CorpusError(f"Could not read source file: {path}") from exc
    except OSError as exc:
        raise CorpusError(f"Could not read source file: {path}") from exc


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised only without extra
        raise CorpusError(
            "PDF support requires the optional dependency set. "
            "Install it with: uv sync --extra pdf (or pip install 'looprag[pdf]')"
        ) from exc
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((number, text))
    return pages


def _title_for(text: str, path: Path) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return path.stem.replace("-", " ").replace("_", " ").strip() or path.name


def _paragraphs(text: str) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = re.split(r"\n\s*\n", normalized)
    return [re.sub(r"\s+", " ", part).strip() for part in parts if part.strip()]


def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 120) -> list[str]:
    """Paragraph-aware chunking with a soft target size and character overlap."""
    if chunk_size <= 0:
        raise CorpusError("chunk_size must be positive")
    chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))
    paragraphs = _paragraphs(text)
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            start = 0
            step = max(1, chunk_size - chunk_overlap)
            while start < len(paragraph):
                piece = paragraph[start : start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                if start + chunk_size >= len(paragraph):
                    break
                start += step
            continue
        if current and len(current) + len(paragraph) + 2 > chunk_size:
            chunks.append(current)
            tail = current[-chunk_overlap:].strip() if chunk_overlap else ""
            if tail and " " in tail:
                tail = tail.split(" ", 1)[1].strip()
            current = tail
        current = f"{current}\n\n{paragraph}" if current else paragraph
    if current.strip():
        chunks.append(current.strip())
    return [chunk for chunk in chunks if chunk.strip()]


def _make_chunk(source: str, title: str, index: int, content: str, page: int | None) -> Chunk:
    digest = hashlib.sha256(f"{source}::{index}::{content}".encode()).hexdigest()[:16]
    chunk: Chunk = {
        "id": digest,
        "source": source,
        "title": title,
        "chunk": index,
        "content": content,
    }
    if page is not None:
        chunk["page"] = page
    return chunk


def load_chunks(source: str | Path, chunk_size: int = 900, chunk_overlap: int = 120) -> list[Chunk]:
    """Load every supported document under ``source`` and chunk it.

    ``source`` metadata is always a relative POSIX path, never absolute.
    """
    path = Path(source)
    if not path.exists():
        raise CorpusError(f"Source not found: {source}")
    if path.is_dir():
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
        )
        root = path
    elif path.suffix.lower() in SUPPORTED_SUFFIXES:
        files = [path]
        root = path.parent
    else:
        raise CorpusError(f"Unsupported source type: {path.suffix or source}")
    chunks: list[Chunk] = []
    for file in files:
        relative = file.relative_to(root).as_posix() if path.is_dir() else file.name
        chunk_index = 0
        if file.suffix.lower() == ".pdf":
            pages = _read_pdf(file)
            title = file.stem.replace("-", " ").replace("_", " ").strip() or file.name
            for page_number, page_text in pages:
                pieces = chunk_text(page_text, chunk_size, chunk_overlap)
                for piece in pieces:
                    chunks.append(_make_chunk(relative, title, chunk_index, piece, page_number))
                    chunk_index += 1
            continue
        text = _read_text(file)
        title = _title_for(text, file)
        for piece in chunk_text(text, chunk_size, chunk_overlap):
            chunks.append(_make_chunk(relative, title, chunk_index, piece, None))
            chunk_index += 1
    if not files:
        raise CorpusError(f"No supported documents found under: {source}")
    return chunks


def make_store(chunks: list[Chunk], chunk_size: int, chunk_overlap: int) -> dict:
    return {
        "version": STORE_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "chunks": chunks,
    }


def save_store(store: dict, path: str | Path) -> Path:
    target = Path(path)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(store, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        raise CorpusError(f"Could not write corpus store: {path}") from exc
    return target


def load_store(path: str | Path) -> list[Chunk]:
    target = Path(path)
    if not target.exists():
        raise CorpusError(
            f"Corpus store not found: {path}. "
            f"Build one first with: looprag ingest <SOURCE> --store {path}"
        )
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except OSError as exc:
        raise CorpusError(f"Could not read corpus store: {path}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CorpusError(f"Corpus store is not valid JSON: {path}") from exc
    if not isinstance(data, dict) or data.get("version") != STORE_VERSION:
        raise CorpusError(f"Unsupported corpus store format in {path}")
    chunks = data.get("chunks")
    if not isinstance(chunks, list):
        raise CorpusError(f"Corpus store has no chunks list: {path}")
    return [chunk for chunk in chunks if isinstance(chunk, dict)]


def _bm25_tokens(chunk: Chunk) -> list[str]:
    text = f"{chunk.get('title', '')} {chunk.get('title', '')} {chunk.get('content', '')}"
    return [
        stem(word)
        for word in tokenize(text)
        if word not in STOPWORDS and (len(word) > 1 or word.isdigit())
    ]


class BM25Retriever:
    """A tiny BM25 retriever over in-memory chunks (no network, no embeddings)."""

    def __init__(self, chunks: list[Chunk], default_k: int = 4) -> None:
        self.chunks = [dict(chunk) for chunk in chunks]
        self.default_k = default_k
        corpus = [_bm25_tokens(chunk) for chunk in self.chunks]
        # BM25Okapi divides by the token count; an all-tokenless corpus (for
        # example only non-ASCII text) must degrade to "no matches" instead.
        self._bm25 = BM25Okapi(corpus) if corpus and any(corpus) else None

    def __len__(self) -> int:
        return len(self.chunks)

    def search(self, query: str, k: int | None = None) -> list[Chunk]:
        k = self.default_k if k is None else k
        if not self.chunks or not query.strip() or k <= 0:
            return []
        tokens = [
            stem(word)
            for word in tokenize(query)
            if word not in STOPWORDS and (len(word) > 1 or word.isdigit())
        ]
        if not tokens or self._bm25 is None:
            return []
        scores = [float(score) for score in self._bm25.get_scores(tokens)]
        order = sorted(
            range(len(self.chunks)),
            key=lambda i: (
                -scores[i],
                self.chunks[i].get("source", ""),
                int(self.chunks[i].get("chunk", 0)),
            ),
        )
        positive = [i for i in order if scores[i] > 0]
        chosen = positive[:k] if positive else order[:k]
        results: list[Chunk] = []
        for index in chosen:
            item = dict(self.chunks[index])
            item["score"] = round(scores[index], 6)
            results.append(item)
        return results
