"""Deterministic offline policy: lexical grading, rewriting, extractive answers.

This is deliberately not an LLM. It is transparent, reproducible, and good
enough to demonstrate the graph's self-correction loop without any API key.
"""

from __future__ import annotations

import re

from .policy import (
    content_terms,
    missing_terms,
    parse_citations,
    sentence_split,
    stem,
    token_coverage,
    tokenize,
    truncate,
)
from .state import Chunk, Critique, Grade

_MISSING_RE = re.compile(r"missing(?:\s+terms)?:\s*([^.;]+)", re.IGNORECASE)
_ISSUE_STOP_WORDS = {
    "answer",
    "answers",
    "citation",
    "citations",
    "claim",
    "claims",
    "content",
    "coverage",
    "document",
    "documents",
    "ground",
    "grounded",
    "invalid",
    "low",
    "match",
    "matched",
    "miss",
    "missing",
    "none",
    "phrase",
    "question",
    "relevance",
    "relevant",
    "source",
    "sources",
    "support",
    "supported",
    "term",
    "terms",
    "unsupported",
}
_ISSUE_STOP = {stem(word) for word in _ISSUE_STOP_WORDS}


class HeuristicPolicy:
    """A small, explainable rule-based policy used in ``--offline`` mode."""

    name = "offline-heuristic"

    def __init__(self, *, relevant_threshold: float = 0.34, cover_threshold: float = 0.55) -> None:
        self.relevant_threshold = relevant_threshold
        self.cover_threshold = cover_threshold

    # ------------------------------------------------------------------ grade
    def grade(self, question: str, docs: list[Chunk]) -> list[Grade]:
        q_terms = content_terms(question)
        q_set = set(q_terms)
        shingles = self._shingles(question)
        grades: list[Grade] = []
        for doc in docs:
            haystack = f"{doc.get('title', '')}\n{doc.get('content', '')}"
            coverage = token_coverage(q_terms, haystack)
            matched = [t for t in q_terms if t not in missing_terms([t], haystack)]
            missing = missing_terms(q_terms, haystack)
            phrase_hit = self._phrase_hit(shingles, haystack)
            relevant = False
            if not q_terms:
                reason = "no content terms in question"
                grades.append({"relevant": False, "reason": reason})
                continue
            if coverage >= self.relevant_threshold:
                relevant = True
            elif len(q_set) <= 2 and matched:
                relevant = True
            elif phrase_hit:
                relevant = True
            reason = f"matched: {', '.join(matched) or 'none'}"
            if missing:
                reason += f"; missing: {', '.join(missing)}"
            if phrase_hit:
                reason += "; phrase match"
            grades.append({"relevant": relevant, "reason": reason})
        return grades

    @staticmethod
    def _shingles(question: str) -> list[str]:
        words = [w for w in tokenize(question) if w not in {"the", "and", "how", "what"}]
        return [" ".join(words[i : i + 3]) for i in range(max(0, len(words) - 2))]

    @staticmethod
    def _phrase_hit(shingles: list[str], haystack: str) -> bool:
        lowered = " ".join(tokenize(haystack))
        return any(shingle and shingle in lowered for shingle in shingles)

    # ---------------------------------------------------------------- rewrite
    def rewrite(self, question: str, query: str, feedback: list[str], attempt: int) -> str:
        terms = content_terms(query)
        seen = set(terms)
        extra: list[str] = []
        for item in feedback:
            for match in _MISSING_RE.finditer(item):
                for piece in re.split(r"[,/]", match.group(1)):
                    term = stem(piece.strip())
                    if term and term not in _ISSUE_STOP and term not in seen:
                        extra.append(term)
                        seen.add(term)
        if not extra:
            for item in feedback:
                for term in content_terms(item):
                    if term.isdigit() or term in _ISSUE_STOP:
                        continue
                    if term not in seen:
                        extra.append(term)
                        seen.add(term)
        combined = terms + extra
        if not combined:
            combined = content_terms(question)
        if not combined:
            combined = tokenize(question)[:12] or ["evidence"]
        # Cap the query; keep the earliest (most important) terms.
        result = " ".join(combined[:12])
        if result == " ".join(terms) and content_terms(question):
            fallback = [t for t in content_terms(question) if t not in seen]
            if fallback:
                result = " ".join((terms + fallback)[:12])
        return result

    # --------------------------------------------------------------- generate
    def generate(self, question: str, docs: list[Chunk], critique: Critique | None) -> str:
        if not docs:
            return "I don't have enough evidence in the corpus to answer that."
        q_terms = content_terms(question)
        q_set = set(q_terms)
        candidates: list[tuple[float, int, int, str]] = []
        fallback: tuple[int, str] | None = None
        for doc_index, doc in enumerate(docs):
            sentences = sentence_split(doc.get("content", ""))
            for sentence_index, sentence in enumerate(sentences):
                if sentence.lstrip().startswith("#"):
                    continue
                if len(sentence) < 25 or len(sentence.split()) < 4:
                    continue
                if fallback is None:
                    fallback = (doc_index, sentence)
                score = token_coverage(q_terms, sentence) * 3.0 + 1.0 / (1.0 + doc_index)
                candidates.append((score, doc_index, sentence_index, sentence))
        selected: list[tuple[int, int, str]] = []
        seen_sentences: set[str] = set()
        covered: set[str] = set()
        remaining = list(candidates)
        while remaining and len(selected) < 4:
            best: tuple[float, int, int, str] | None = None
            best_key: tuple[float, int, int] | None = None
            for item in remaining:
                sentence_terms = set(content_terms(item[3]))
                gain = len((q_set - covered) & sentence_terms)
                key = (item[0] + 0.8 * gain, -item[1], -item[2])
                if best_key is None or key > best_key:
                    best = item
                    best_key = key
            assert best is not None
            remaining.remove(best)
            sentence_key = " ".join(tokenize(best[3]))
            if sentence_key in seen_sentences:
                continue
            seen_sentences.add(sentence_key)
            selected.append((best[1], best[2], best[3]))
            covered |= set(content_terms(best[3]))
        if not selected and fallback is not None:
            selected = [(fallback[0], 0, fallback[1])]
        if not selected:
            top = docs[0]
            text = truncate(top.get("content", ""), 300)
            if not text:
                return "I don't have enough evidence in the corpus to answer that."
            return f"{text} [1]"
        selected.sort(key=lambda item: (item[0], item[1]))
        parts: list[str] = []
        for doc_index, _, sentence in selected:
            clean = sentence.strip()
            if clean[-1] not in ".!?":
                clean += "."
            parts.append(f"{clean} [{doc_index + 1}]")
        return " ".join(parts)

    # ----------------------------------------------------------------- verify
    def verify(self, question: str, answer: str, docs: list[Chunk]) -> Critique:
        issues: list[str] = []
        unsupported: list[str] = []
        prose = re.sub(r"\[\d+\]", "", answer)
        answer_terms = content_terms(prose)
        bundle = "\n".join(f"{doc.get('title', '')}\n{doc.get('content', '')}" for doc in docs)
        coverage = token_coverage(answer_terms, bundle)
        citations = parse_citations(answer)
        invalid = [n for n in citations if not 1 <= n <= len(docs)]
        if invalid:
            issues.append("invalid citations: " + ", ".join(f"[{n}]" for n in invalid))
        missing = missing_terms(answer_terms, bundle)
        if missing:
            issues.append("missing terms: " + ", ".join(missing[:8]))
        lowered = answer.lower()
        insufficient = "not enough evidence" in lowered or "insufficient evidence" in lowered
        if insufficient:
            issues.append("answer reports insufficient evidence")
        if not docs:
            issues.append("no source documents")
        for sentence in sentence_split(answer):
            if len(sentence.split()) < 5:
                continue
            sentence_terms = content_terms(re.sub(r"\[\d+\]", "", sentence))
            if sentence_terms and token_coverage(sentence_terms, bundle) < 0.5:
                unsupported.append(truncate(sentence, 120))
        grounded = (
            bool(docs)
            and not invalid
            and coverage >= self.cover_threshold
            and not insufficient
            and not unsupported
        )
        if unsupported:
            issues.append(f"unsupported claims: {len(unsupported)}")
        if not grounded and coverage < self.cover_threshold and not missing:
            issues.append(f"low source coverage: {coverage:.2f}")
        return {"grounded": grounded, "issues": issues, "unsupported_claims": unsupported[:3]}
