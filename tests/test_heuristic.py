"""Heuristic policy behaviour: tokenization, grading, rewriting and verification."""

from __future__ import annotations

from looprag.heuristic import HeuristicPolicy
from looprag.policy import content_terms, parse_citations, stem, token_coverage, tokenize
from looprag.state import Chunk


def _chunk(content: str, source: str = "doc.md", title: str = "Doc") -> Chunk:
    return {
        "id": "abc123",
        "source": source,
        "title": title,
        "chunk": 0,
        "content": content,
        "score": 1.0,
    }


POWER_DOC = _chunk(
    "The station is powered by a micro-hydro turbine in summer and diesel generators in "
    "winter. During a power emergency the controller sheds loads and starts the emergency "
    "generator. The battery bank stores roughly fifty-seven kilowatt-hours.",
    source="power-system.md",
    title="Power System",
)

COOKING_DOC = _chunk(
    "The cook prepares breakfast at seven and dinner at half past six. The kitchen "
    "inventory is checked every Friday.",
    source="station-overview.md",
    title="Station Overview",
)

QUESTION = "How is the station powered in winter, and what happens during a power emergency?"


def test_tokenization_and_stemming() -> None:
    assert tokenize("Runs RUNNING foxes!") == ["runs", "running", "foxes"]
    assert stem("running") == "run"
    assert stem("powered") == "power"
    assert stem("monitoring") == "monitor"
    assert stem("foxes") == "fox"
    assert stem("batteries") == "battery"
    assert "the" not in content_terms("The station and the battery")
    assert content_terms("The station and the battery") == ["station", "battery"]


def test_token_coverage() -> None:
    assert token_coverage(["power", "winter"], "winter power systems") == 1.0
    assert token_coverage(["power", "diesel"], "winter power systems") == 0.5
    assert token_coverage([], "anything") == 0.0


def test_grade_relevant_and_irrelevant() -> None:
    policy = HeuristicPolicy()
    grades = policy.grade(QUESTION, [POWER_DOC, COOKING_DOC])
    assert grades[0]["relevant"] is True
    assert "matched:" in grades[0]["reason"]
    assert grades[1]["relevant"] is False
    assert "missing:" in grades[1]["reason"]


def test_grade_short_question_matches_single_term() -> None:
    policy = HeuristicPolicy()
    grades = policy.grade("power emergency", [POWER_DOC])
    assert grades[0]["relevant"] is True


def test_rewrite_adds_missing_terms_deterministically() -> None:
    policy = HeuristicPolicy()
    query = policy.rewrite(
        QUESTION,
        "station power",
        ["matched: station; missing: emergency, diesel"],
        1,
    )
    assert "emergency" in query
    assert "diesel" in query
    assert query == policy.rewrite(
        QUESTION,
        "station power",
        ["matched: station; missing: emergency, diesel"],
        1,
    )
    unchanged = policy.rewrite(QUESTION, "station power", [], 1)
    assert unchanged.strip()
    assert unchanged == unchanged.strip()


def test_rewrite_ignores_structural_feedback_words_and_citation_digits() -> None:
    policy = HeuristicPolicy()
    assert (
        policy.rewrite(
            "What colour is the submarine?", "colour submarine", ["invalid citations: [9]"], 1
        )
        == "colour submarine"
    )
    assert (
        policy.rewrite(
            "How is the station powered?",
            "station power winter",
            ["answer not grounded; missing terms: station"],
            1,
        )
        == "station power winter"
    )
    query = policy.rewrite("what is it?", "x", ["content term question"], 1)
    assert query == "what is it"
    assert "content" not in query.split()
    assert "question" not in query.split()


def test_generate_cites_valid_sources() -> None:
    policy = HeuristicPolicy()
    answer = policy.generate(QUESTION, [POWER_DOC], None)
    assert answer
    citations = parse_citations(answer)
    assert citations
    assert all(1 <= number <= 1 for number in citations)
    assert "[1]" in answer


def test_generate_without_documents_is_honest() -> None:
    answer = HeuristicPolicy().generate(QUESTION, [], None)
    assert "enough evidence" in answer.lower()


def test_verify_rejects_fabricated_claim() -> None:
    policy = HeuristicPolicy()
    critique = policy.verify(
        QUESTION, "The station is powered by a nuclear reactor [1].", [POWER_DOC]
    )
    assert critique["grounded"] is False
    assert critique["issues"]


def test_verify_rejects_invalid_citation() -> None:
    policy = HeuristicPolicy()
    critique = policy.verify(
        QUESTION,
        "The station is powered by diesel generators in winter [9].",
        [POWER_DOC],
    )
    assert critique["grounded"] is False
    assert any("invalid citations" in issue for issue in critique["issues"])


def test_verify_accepts_extractive_answer() -> None:
    policy = HeuristicPolicy()
    answer = (
        "The station is powered by a micro-hydro turbine in summer and diesel generators "
        "in winter [1]."
    )
    critique = policy.verify(QUESTION, answer, [POWER_DOC])
    assert critique["grounded"] is True
    assert isinstance(critique["issues"], list)


def test_verify_citation_markers_are_not_content_terms() -> None:
    policy = HeuristicPolicy()
    short = policy.verify("How is the station powered in winter?", "Diesel [1].", [POWER_DOC])
    assert short["grounded"] is True
    assert short["issues"] == []
    long = policy.verify(
        QUESTION,
        "The station is powered by diesel generators in winter [1].",
        [POWER_DOC],
    )
    assert long["grounded"] is True
    assert not any("missing terms" in issue for issue in long["issues"])


def test_verify_rejects_mixed_supported_and_fabricated_answer() -> None:
    policy = HeuristicPolicy()
    answer = (
        "The station is powered by a micro-hydro turbine in summer and diesel generators "
        "in winter [1]. Fusion reactors supply power [1]."
    )
    critique = policy.verify(QUESTION, answer, [POWER_DOC])
    assert critique["grounded"] is False
    assert critique["unsupported_claims"]
    assert any("unsupported claims" in issue for issue in critique["issues"])


def test_verify_refuses_insufficient_answer() -> None:
    policy = HeuristicPolicy()
    critique = policy.verify(
        QUESTION, "I don't have enough evidence in the corpus to answer that.", [POWER_DOC]
    )
    assert critique["grounded"] is False
