"""Graph topology tests with a scripted policy and stub retriever."""

from __future__ import annotations

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from looprag.graph import build_graph, run_config
from looprag.state import Chunk, Critique


def _chunk(content: str, source: str = "doc.md") -> Chunk:
    return {
        "id": content[:8],
        "source": source,
        "title": source,
        "chunk": 0,
        "content": content,
        "score": 1.0,
    }


IRRELEVANT = _chunk("The kitchen inventory is checked on Fridays.", "kitchen.md")
RELEVANT = _chunk("The needle is blue and belongs to the winter kit.", "needle.md")


class StubRetriever:
    default_k = 4

    def __init__(self, mapping: dict[str, list[Chunk]], fallback: list[Chunk] | None = None):
        self.mapping = mapping
        self.fallback = fallback or []
        self.calls: list[str] = []

    def search(self, query: str, k: int | None = 4) -> list[Chunk]:
        self.calls.append(query)
        docs = self.mapping.get(query, self.fallback)
        return [dict(doc) for doc in docs[: (k or 4)]]


class ScriptedPolicy:
    name = "scripted"

    def __init__(
        self,
        *,
        relevant: Any = lambda question, doc: True,
        answers: list[str] | None = None,
        critiques: list[Critique] | None = None,
        rewrites: dict[str, str] | None = None,
    ) -> None:
        self.relevant = relevant
        self.answers = list(answers or [])
        self.critiques = list(critiques or [])
        self.rewrites = rewrites or {}
        self.rewrite_calls = 0
        self.grade_calls = 0

    def grade(self, question: str, docs: list[Chunk]) -> list[dict]:
        self.grade_calls += 1
        result = []
        for doc in docs:
            is_relevant = bool(self.relevant(question, doc))
            result.append(
                {
                    "relevant": is_relevant,
                    "reason": "relevant" if is_relevant else "missing: needle",
                }
            )
        return result

    def rewrite(self, question: str, query: str, feedback: list[str], attempt: int) -> str:
        self.rewrite_calls += 1
        if query in self.rewrites:
            return self.rewrites[query]
        return f"{query} retry{attempt + 1}"

    def generate(self, question: str, docs: list[Chunk], critique: Critique | None) -> str:
        assert self.answers, "generate called more times than scripted"
        return self.answers.pop(0)

    def verify(self, question: str, answer: str, docs: list[Chunk]) -> Critique:
        assert self.critiques, "verify called more times than scripted"
        return self.critiques.pop(0)


def _initial(question: str, query: str | None = None) -> dict:
    return {
        "question": question,
        "query": query or question,
        "attempt": 0,
        "review_round": 0,
        "trace": [],
    }


def _nodes(state: dict) -> list[str]:
    return [event["node"] for event in state.get("trace", [])]


def _run(graph, question: str, query: str | None = None, thread: str = "t") -> dict:
    config = run_config(thread)
    graph.invoke(_initial(question, query), config)
    return dict(graph.get_state(config).values)


def test_happy_path_exact_node_order() -> None:
    policy = ScriptedPolicy(
        answers=["The station uses diesel [1]."],
        critiques=[{"grounded": True, "issues": [], "unsupported_claims": []}],
    )
    graph = build_graph(
        policy, StubRetriever({"q1": [RELEVANT]}), max_attempts=2, checkpointer=InMemorySaver()
    )
    state = _run(graph, "q1", thread="happy")
    assert _nodes(state) == ["retrieve", "grade", "generate", "verify", "finalize"]
    assert state["attempt"] == 1
    assert state["grounded"] is True
    assert state["final"] == "The station uses diesel [1]."
    assert state["insufficient"] is False


def test_rewrite_path_changes_query_and_grades_twice() -> None:
    policy = ScriptedPolicy(
        relevant=lambda question, doc: "needle" in doc["content"],
        answers=["The needle is blue [1]."],
        critiques=[{"grounded": True, "issues": [], "unsupported_claims": []}],
        rewrites={"q1": "q1 fixed"},
    )
    retriever = StubRetriever({"q1": [IRRELEVANT], "q1 fixed": [RELEVANT]})
    graph = build_graph(policy, retriever, max_attempts=2, checkpointer=InMemorySaver())
    state = _run(graph, "q1", thread="rewrite")
    assert _nodes(state) == [
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
        "verify",
        "finalize",
    ]
    assert _nodes(state).count("grade") == 2
    assert state["attempt"] == 2
    assert state["grounded"] is True
    assert "[1]" in state["final"]
    assert retriever.calls == ["q1", "q1 fixed"]
    rewrite_events = [event for event in state["trace"] if event["node"] == "rewrite"]
    assert "q1 fixed" in rewrite_events[0]["message"]


def test_exhausted_retrieval_is_bounded_and_insufficient() -> None:
    policy = ScriptedPolicy(relevant=lambda question, doc: False)
    retriever = StubRetriever({}, fallback=[IRRELEVANT])
    graph = build_graph(policy, retriever, max_attempts=2, checkpointer=InMemorySaver())
    state = _run(graph, "q1", thread="exhausted")
    assert _nodes(state) == [
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "insufficient",
    ]
    assert state["attempt"] == 2
    assert state["insufficient"] is True
    assert state["grounded"] is False
    assert "enough evidence" in state["final"].lower()
    assert len(_nodes(state)) <= 10
    assert policy.grade_calls == 2


def test_verification_retry_rewrites_once_then_finalizes() -> None:
    policy = ScriptedPolicy(
        answers=["Unsupported claim [9].", "The needle is blue [1]."],
        critiques=[
            {
                "grounded": False,
                "issues": ["invalid citation [9]"],
                "unsupported_claims": ["Unsupported claim"],
            },
            {"grounded": True, "issues": [], "unsupported_claims": []},
        ],
    )
    graph = build_graph(
        policy, StubRetriever({}, fallback=[RELEVANT]), max_attempts=2, checkpointer=InMemorySaver()
    )
    state = _run(graph, "q1", thread="verify-retry")
    assert _nodes(state) == [
        "retrieve",
        "grade",
        "generate",
        "verify",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
        "verify",
        "finalize",
    ]
    assert state["attempt"] == 2
    assert state["grounded"] is True
    assert state["final"] == "The needle is blue [1]."
    assert policy.rewrite_calls == 1


def test_high_attempt_budget_terminates_within_recursion_limit() -> None:
    policy = ScriptedPolicy(
        answers=[f"Draft {index} [1]." for index in range(8)],
        critiques=[
            {"grounded": False, "issues": [f"unsupported {index}"], "unsupported_claims": []}
            for index in range(8)
        ],
    )
    retriever = StubRetriever({}, fallback=[RELEVANT])
    graph = build_graph(policy, retriever, max_attempts=8, checkpointer=InMemorySaver())
    config = run_config("budget")
    assert config["recursion_limit"] == 60
    graph.invoke(_initial("q1"), config)
    state = dict(graph.get_state(config).values)
    assert state["attempt"] == 8
    assert state["grounded"] is False
    assert _nodes(state)[-1] == "finalize"
    assert len(_nodes(state)) <= 60


def test_stream_updates_can_be_consumed() -> None:
    policy = ScriptedPolicy(
        answers=["Answer one [1]."],
        critiques=[{"grounded": True, "issues": [], "unsupported_claims": []}],
    )
    graph = build_graph(
        policy, StubRetriever({"q1": [RELEVANT]}), max_attempts=2, checkpointer=InMemorySaver()
    )
    config = run_config("stream")
    frames = list(graph.stream(_initial("q1"), config, stream_mode="updates"))
    seen = [node for frame in frames for node in frame if not node.startswith("__")]
    assert seen == ["retrieve", "grade", "generate", "verify", "finalize"]
    final_state = graph.get_state(config).values
    assert final_state["final"] == "Answer one [1]."


def test_review_without_checkpointer_is_rejected() -> None:
    policy = ScriptedPolicy()
    with pytest.raises(ValueError, match="checkpointer"):
        build_graph(policy, StubRetriever({}), enable_review=True, checkpointer=None)
