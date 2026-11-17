"""Typed state shared by every LoopRAG graph node."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class Chunk(TypedDict, total=False):
    """A retrieved chunk. Stored chunks omit ``score`` until retrieval."""

    id: str
    source: str
    title: str
    content: str
    chunk: int
    page: int
    score: float


class Grade(TypedDict):
    relevant: bool
    reason: str


class Critique(TypedDict):
    grounded: bool
    issues: list[str]
    unsupported_claims: list[str]


class TraceEvent(TypedDict):
    node: str
    message: str


class LoopRAGState(TypedDict, total=False):
    question: str
    query: str
    attempt: int
    documents: list[Chunk]
    grades: list[Grade]
    relevant: list[Chunk]
    answer: str
    critique: Critique | None
    grounded: bool
    final: str
    insufficient: bool
    review_round: int
    approved: bool | None
    notice: str | None
    trace: Annotated[list[TraceEvent], operator.add]
