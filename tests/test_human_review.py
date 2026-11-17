"""Human-in-the-loop review: interrupts, approvals, rejections and budgets."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from looprag.graph import build_graph, run_config
from looprag.state import Chunk, Critique


def _chunk(content: str, source: str = "doc.md") -> Chunk:
    return {
        "id": "c1",
        "source": source,
        "title": source,
        "chunk": 0,
        "content": content,
        "score": 1.0,
    }


DOC = _chunk("The needle is blue and belongs to the winter kit.", "needle.md")


class StubRetriever:
    default_k = 4

    def __init__(self, docs: list[Chunk]):
        self.docs = docs

    def search(self, query: str, k: int | None = 4) -> list[Chunk]:
        return [dict(doc) for doc in self.docs[: (k or 4)]]


class ScriptedPolicy:
    name = "scripted"

    def __init__(self, answers: list[str], critiques: list[Critique]) -> None:
        self.answers = list(answers)
        self.critiques = list(critiques)

    def grade(self, question: str, docs: list[Chunk]) -> list[dict]:
        return [{"relevant": True, "reason": "relevant"} for _ in docs]

    def rewrite(self, question: str, query: str, feedback: list[str], attempt: int) -> str:
        return f"{query} revised"

    def generate(self, question: str, docs: list[Chunk], critique: Critique | None) -> str:
        return self.answers.pop(0)

    def verify(self, question: str, answer: str, docs: list[Chunk]) -> Critique:
        return self.critiques.pop(0)


def _initial(question: str) -> dict:
    return {
        "question": question,
        "query": question,
        "attempt": 0,
        "review_round": 0,
        "trace": [],
    }


def _payload(result: dict) -> dict:
    interrupts = result["__interrupt__"]
    return interrupts[0].value


def _nodes(state: dict) -> list[str]:
    return [event["node"] for event in state.get("trace", [])]


def _graph(policy: Any, *, max_reviews: int = 1, max_attempts: int = 2):
    return build_graph(
        policy,
        StubRetriever([DOC]),
        max_attempts=max_attempts,
        enable_review=True,
        max_reviews=max_reviews,
        checkpointer=InMemorySaver(),
    )


def _grounded() -> Critique:
    return {"grounded": True, "issues": [], "unsupported_claims": []}


def test_interrupt_exposes_draft_sources_and_pending_node() -> None:
    policy = ScriptedPolicy(["Draft one [1]."], [_grounded()])
    graph = _graph(policy)
    config = run_config("review-interrupt")
    result = graph.invoke(_initial("What colour is the needle?"), config)
    assert "__interrupt__" in result
    payload = _payload(result)
    assert payload["draft"] == "Draft one [1]."
    assert payload["question"] == "What colour is the needle?"
    assert payload["allowed"] == {"approve": True, "reject": True}
    assert payload["grounded"] is True
    assert payload["sources"][0]["source"] == "needle.md"
    assert graph.get_state(config).next == ("review",)


def test_resume_approve_finishes() -> None:
    policy = ScriptedPolicy(["Draft one [1]."], [_grounded()])
    graph = _graph(policy)
    config = run_config("review-approve")
    graph.invoke(_initial("q"), config)
    result = graph.invoke(Command(resume={"approved": True, "feedback": ""}), config)
    assert "__interrupt__" not in result
    state = dict(graph.get_state(config).values)
    assert state["approved"] is True
    assert state["review_round"] == 1
    assert state["final"] == "Draft one [1]."
    assert _nodes(state) == ["retrieve", "grade", "generate", "verify", "finalize", "review"]
    assert "approved" in state["trace"][-1]["message"]
    assert graph.get_state(config).next == ()


def test_reject_feedback_rewrites_then_second_review() -> None:
    policy = ScriptedPolicy(
        ["Draft one [1].", "Draft two [1]."],
        [_grounded(), _grounded()],
    )
    graph = _graph(policy, max_reviews=2, max_attempts=2)
    config = run_config("review-reject")
    graph.invoke(_initial("q"), config)
    result = graph.invoke(
        Command(resume={"approved": False, "feedback": "mention the winter kit"}), config
    )
    assert "__interrupt__" in result
    payload = _payload(result)
    assert payload["review_round"] == 2
    assert payload["draft"] == "Draft two [1]."
    state = dict(graph.get_state(config).values)
    assert state["approved"] is False
    first_cycle = [
        "retrieve",
        "grade",
        "generate",
        "verify",
        "finalize",
        "review",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
        "verify",
        "finalize",
    ]
    assert _nodes(state) == first_cycle
    assert _nodes(state).count("review") == 1
    final = graph.invoke(Command(resume={"approved": True, "feedback": ""}), config)
    assert "__interrupt__" not in final
    state = dict(graph.get_state(config).values)
    assert _nodes(state) == [*first_cycle, "review"]
    assert state["review_round"] == 2
    assert state["approved"] is True
    assert state["final"] == "Draft two [1]."


def test_reject_with_exhausted_budget_annotates_final() -> None:
    policy = ScriptedPolicy(["Draft one [1]."], [_grounded()])
    graph = _graph(policy, max_reviews=1)
    config = run_config("review-exhausted")
    graph.invoke(_initial("q"), config)
    graph.invoke(Command(resume={"approved": False, "feedback": "not convincing"}), config)
    state = dict(graph.get_state(config).values)
    assert state["approved"] is False
    assert state["review_round"] == 1
    assert "rejected" in state["final"].lower()
    assert "not convincing" in state["final"]
    assert state["notice"]
    assert _nodes(state) == ["retrieve", "grade", "generate", "verify", "finalize", "review"]
    assert "exhausted" in state["trace"][-1]["message"]


def test_missing_decision_defaults_to_reject() -> None:
    policy = ScriptedPolicy(["Draft one [1]."], [_grounded()])
    graph = _graph(policy, max_reviews=1)
    config = run_config("review-default-reject")
    graph.invoke(_initial("q"), config)
    graph.invoke(Command(resume={"feedback": "no thanks"}), config)
    state = dict(graph.get_state(config).values)
    assert state["approved"] is False
    assert "No decision provided" in state["final"]
    assert _nodes(state) == ["retrieve", "grade", "generate", "verify", "finalize", "review"]


def test_reject_with_empty_feedback_is_not_reported_as_missing_decision() -> None:
    policy = ScriptedPolicy(["Draft one [1]."], [_grounded()])
    graph = _graph(policy, max_reviews=1)
    config = run_config("review-empty-feedback")
    graph.invoke(_initial("q"), config)
    graph.invoke(Command(resume={"approved": False, "feedback": ""}), config)
    state = dict(graph.get_state(config).values)
    assert state["approved"] is False
    assert "no feedback given" in state["final"]
    assert "No decision provided" not in state["final"]
    assert _nodes(state) == ["retrieve", "grade", "generate", "verify", "finalize", "review"]
