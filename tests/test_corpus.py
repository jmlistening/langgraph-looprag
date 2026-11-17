"""Corpus loading, chunking, store round-trip and BM25 retrieval tests."""

from __future__ import annotations

import json
from pathlib import Path

from looprag.corpus import (
    BM25Retriever,
    chunk_text,
    load_chunks,
    load_store,
    make_store,
    save_store,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS = REPO_ROOT / "examples" / "corpus"


def test_load_chunks_uses_relative_stable_ids() -> None:
    chunks = load_chunks(CORPUS)
    assert chunks
    sources = {chunk["source"] for chunk in chunks}
    assert "power-system.md" in sources
    for chunk in chunks:
        assert not Path(chunk["source"]).is_absolute()
        assert "\\" not in chunk["source"]
        assert len(chunk["id"]) == 16
        assert chunk["content"]
        assert isinstance(chunk["chunk"], int)
    ids = [chunk["id"] for chunk in chunks]
    assert len(ids) == len(set(ids))
    again = load_chunks(CORPUS)
    assert [chunk["id"] for chunk in again] == ids


def test_chunk_indices_are_contiguous_per_source() -> None:
    chunks = load_chunks(CORPUS)
    by_source: dict[str, list[int]] = {}
    for chunk in chunks:
        by_source.setdefault(chunk["source"], []).append(chunk["chunk"])
    for indices in by_source.values():
        assert indices == list(range(len(indices)))


def test_chunk_text_keeps_paragraphs_and_size() -> None:
    text = "\n\n".join(
        [
            "Alpha paragraph about the hydro turbine and its intake screen.",
            "Beta paragraph about the battery bank and winter diesel backup.",
            "Gamma paragraph about acoustic sensors and calibration intervals.",
        ]
    )
    chunks = chunk_text(text, chunk_size=110, chunk_overlap=30)
    assert len(chunks) >= 2
    joined = " ".join(chunks)
    for word in ("Alpha", "Beta", "Gamma"):
        assert word in joined
    assert max(len(chunk) for chunk in chunks) <= 110 + 30 + 10


def test_chunk_text_handles_oversized_paragraph() -> None:
    text = "word " * 300
    chunks = chunk_text(text.strip(), chunk_size=200, chunk_overlap=40)
    assert len(chunks) > 1
    assert all(len(chunk) <= 200 for chunk in chunks)


def test_store_round_trip(tmp_path: Path) -> None:
    chunks = load_chunks(CORPUS)
    store = make_store(chunks, 900, 120)
    target = tmp_path / "nested" / "corpus.json"
    save_store(store, target)
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["chunk_size"] == 900
    assert raw["chunk_overlap"] == 120
    assert "created_at" in raw
    loaded = load_store(target)
    assert loaded == chunks
    text = target.read_text(encoding="utf-8")
    assert "C:" + "/Users" not in text
    assert ":\\" not in text


def test_bm25_ranks_power_document_first() -> None:
    retriever = BM25Retriever(load_chunks(CORPUS))
    results = retriever.search("diesel generator micro-hydro power emergency", k=4)
    assert results
    assert results[0]["source"] == "power-system.md"
    scores = [result["score"] for result in results]
    assert scores == sorted(scores, reverse=True)
    assert all("title" in result for result in results)


def test_bm25_spans_multiple_documents() -> None:
    retriever = BM25Retriever(load_chunks(CORPUS))
    results = retriever.search("permits drones acoustic calibration visitors", k=4)
    assert len({result["source"] for result in results}) >= 2


def test_bm25_zero_match_falls_back_without_crashing() -> None:
    retriever = BM25Retriever(load_chunks(CORPUS))
    results = retriever.search("zebra quantum xylophone", k=2)
    assert len(results) <= 2
    assert all(result["score"] <= 0.0 for result in results)
    assert retriever.search("", k=2) == []
    assert len(BM25Retriever([]).search("power")) == 0


def test_bm25_tokenless_corpus_degrades_to_no_matches() -> None:
    chunks = [
        {
            "id": "a" * 16,
            "source": "zh.md",
            "title": "站",
            "chunk": 0,
            "content": "站的中文内容",
        },
        {
            "id": "b" * 16,
            "source": "stop.md",
            "title": "the",
            "chunk": 0,
            "content": "the and of to",
        },
    ]
    retriever = BM25Retriever(chunks)
    assert retriever.search("anything") == []
    assert retriever.search("中文") == []
