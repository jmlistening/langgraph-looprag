"""Policy protocol and shared text utilities.

The helpers here are intentionally small and dependency-free: both the offline
heuristic policy and the tests use the same tokenization, stemming and coverage
functions so behaviour stays predictable.
"""

from __future__ import annotations

import re
from typing import Protocol

from .state import Chunk, Critique, Grade

WORD_RE = re.compile(r"[a-z0-9][a-z0-9'-]*")

STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "being",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "for",
        "from",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "out",
        "over",
        "own",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "to",
        "too",
        "under",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "about",
        "also",
        "any",
        "because",
        "during",
        "each",
        "other",
        "only",
        "same",
        "through",
        "between",
    }
)


class PolicyError(RuntimeError):
    """Raised when the online policy cannot complete a request."""


def tokenize(text: str) -> list[str]:
    """Lowercase regex words; hyphens and apostrophes are kept inside words."""
    return WORD_RE.findall(text.lower())


def stem(word: str) -> str:
    """A deliberately conservative light stemmer.

    Handles common plurals and the -ing/-ed forms. It is good enough for
    lexical retrieval and stable enough to explain in a README.
    """
    if len(word) <= 3 or not word.isalpha():
        return word
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "y"
    elif word.endswith("sses") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("es") and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word = word[:-1]
    if word.endswith("ing") and len(word) - 3 >= 3:
        word = word[:-3]
    elif word.endswith("ed") and len(word) - 2 >= 3:
        word = word[:-2]
    if len(word) >= 4 and word[-1] == word[-2] and word[-1] not in "lsz":
        word = word[:-1]
    return word


def content_terms(text: str) -> list[str]:
    """Unique content stems in first-seen order (stopwords removed)."""
    seen: dict[str, None] = {}
    for raw in tokenize(text):
        if raw in STOPWORDS:
            continue
        term = stem(raw)
        if len(term) < 2 and not term.isdigit():
            continue
        seen.setdefault(term, None)
    return list(seen)


def token_coverage(needles: list[str] | set[str], haystack: str) -> float:
    """Fraction of unique needle terms that occur in ``haystack``."""
    terms = {stem(t) for t in needles if t and t not in STOPWORDS}
    terms = {t for t in terms if t}
    if not terms:
        return 0.0
    words = {stem(w) for w in tokenize(haystack)}
    hit = sum(1 for term in terms if term in words)
    return hit / len(terms)


def missing_terms(needles: list[str] | set[str], haystack: str) -> list[str]:
    """Needle terms (stems) that do not occur in ``haystack``, in input order."""
    words = {stem(w) for w in tokenize(haystack)}
    missing: list[str] = []
    for term in needles:
        if term and term not in words and term not in missing:
            missing.append(term)
    return missing


def parse_citations(answer: str) -> list[int]:
    """Return every ``[n]`` citation number in order of appearance."""
    return [int(match) for match in re.findall(r"\[(\d+)\]", answer)]


def sentence_split(text: str) -> list[str]:
    """Split prose into sentence-ish units, keeping terminal punctuation."""
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


class Policy(Protocol):
    """The four decisions LoopRAG needs from an answer engine."""

    name: str

    def grade(self, question: str, docs: list[Chunk]) -> list[Grade]: ...

    def rewrite(self, question: str, query: str, feedback: list[str], attempt: int) -> str: ...

    def generate(self, question: str, docs: list[Chunk], critique: Critique | None) -> str: ...

    def verify(self, question: str, answer: str, docs: list[Chunk]) -> Critique: ...
