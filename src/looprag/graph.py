"""The LoopRAG state graph: retrieve → grade → generate → verify, with bounded
rewrite cycles and an optional human review interrupt.
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .policy import truncate
from .state import Critique, LoopRAGState, TraceEvent


def _event(node: str, message: str) -> dict[str, list[TraceEvent]]:
    return {"trace": [{"node": node, "message": message}]}


def build_graph(
    policy: Any,
    retriever: Any,
    *,
    max_attempts: int = 2,
    enable_review: bool = False,
    max_reviews: int = 1,
    checkpointer: Any = None,
):
    """Build and compile the LoopRAG graph.

    The checkpointer is attached here when supplied; the CLI owns its
    lifecycle (and its SQLite connection). Human review requires one.
    """
    # Review rounds add cycles; keep the combined worst case under recursion_limit=60.
    max_attempts = max(1, min(int(max_attempts), 5 if enable_review else 8))
    max_reviews = max(0, min(int(max_reviews), 4))
    if enable_review and checkpointer is None:
        raise ValueError(
            "Human review requires a checkpointer. Build the graph with "
            "checkpointer=InMemorySaver() or use --sqlite PATH."
        )

    def retrieve(state: LoopRAGState) -> dict:
        query = state.get("query") or state.get("question", "")
        if hasattr(retriever, "default_k"):
            docs = retriever.search(query, k=retriever.default_k)
        else:
            docs = retriever.search(query)
        docs = list(docs)
        message = f"{len(docs)} chunks for query '{truncate(query, 60)}'"
        return {"documents": docs, "query": query, **_event("retrieve", message)}

    def grade(state: LoopRAGState) -> dict:
        attempt = int(state.get("attempt", 0)) + 1
        docs = list(state.get("documents", []))
        grades = list(policy.grade(state.get("question", ""), docs)) if docs else []
        if len(grades) < len(docs):
            grades.extend(
                {"relevant": False, "reason": "no grade returned"}
                for _ in range(len(docs) - len(grades))
            )
        relevant = [
            doc for doc, verdict in zip(docs, grades, strict=False) if verdict.get("relevant")
        ]
        message = f"{len(relevant)}/{len(docs)} relevant (attempt {attempt})"
        return {
            "attempt": attempt,
            "grades": grades,
            "relevant": relevant,
            **_event("grade", message),
        }

    def rewrite(state: LoopRAGState) -> dict:
        question = state.get("question", "")
        old_query = state.get("query") or question
        feedback: list[str] = []
        for verdict in state.get("grades", []):
            if not verdict.get("relevant"):
                feedback.append(str(verdict.get("reason", "")))
        critique = state.get("critique")
        if critique:
            feedback.extend(str(item) for item in critique.get("issues", []))
            feedback.extend(str(item) for item in critique.get("unsupported_claims", []))
        new_query = policy.rewrite(question, old_query, feedback, int(state.get("attempt", 0)))
        new_query = (new_query or "").strip()
        if not new_query:
            new_query = old_query
        message = f"query '{truncate(old_query, 40)}' -> '{truncate(new_query, 40)}'"
        return {"query": new_query, **_event("rewrite", message)}

    def generate(state: LoopRAGState) -> dict:
        docs = list(state.get("relevant") or state.get("documents", []))
        answer = policy.generate(state.get("question", ""), docs, state.get("critique"))
        return {"answer": answer, **_event("generate", "answer drafted")}

    def verify(state: LoopRAGState) -> dict:
        docs = list(state.get("relevant") or state.get("documents", []))
        critique = policy.verify(state.get("question", ""), state.get("answer", ""), docs)
        grounded = bool(critique.get("grounded"))
        if grounded:
            message = "grounded"
        else:
            issues = critique.get("issues") or []
            detail = truncate(issues[0], 70) if issues else "no supporting evidence"
            message = f"not grounded ({detail})"
        return {
            "critique": critique,
            "grounded": grounded,
            **_event("verify", message),
        }

    def insufficient(state: LoopRAGState) -> dict:
        attempts = int(state.get("attempt", 0))
        return {
            "final": (
                "I don't have enough evidence in the corpus to answer that. "
                f"Tried {attempts} retrieval attempt(s) with no relevant documents."
            ),
            "insufficient": True,
            "grounded": False,
            **_event("insufficient", f"no relevant evidence after {attempts} attempt(s)"),
        }

    def finalize(state: LoopRAGState) -> dict:
        answer = state.get("answer", "")
        grounded = bool(state.get("grounded"))
        notice = None if grounded else "Answer could not be fully verified against sources."
        message = "done" if grounded else "done with unverified answer"
        return {
            "final": answer,
            "notice": notice,
            "insufficient": bool(state.get("insufficient", False)),
            **_event("finalize", message),
        }

    def review(state: LoopRAGState) -> dict:
        review_round = int(state.get("review_round", 0)) + 1
        docs = list(state.get("relevant") or state.get("documents", []))
        sources = [
            {
                "n": number,
                "source": doc.get("source", "?"),
                "snippet": truncate(doc.get("content", ""), 160),
            }
            for number, doc in enumerate(docs, start=1)
        ]
        payload = {
            "question": state.get("question", ""),
            "draft": state.get("answer", ""),
            "sources": sources,
            "attempt": int(state.get("attempt", 0)),
            "grounded": bool(state.get("grounded")),
            "review_round": review_round,
            "allowed": {"approve": True, "reject": True},
        }
        decision = interrupt(payload)
        approved: bool | None = None
        feedback = ""
        if isinstance(decision, dict):
            candidate = decision.get("approved")
            if isinstance(candidate, bool):
                approved = candidate
                raw_feedback = decision.get("feedback", "")
                if isinstance(raw_feedback, str):
                    feedback = raw_feedback.strip()
        if approved is None:
            approved = False
            feedback = "No decision provided"
        if approved:
            return {
                "approved": True,
                "review_round": review_round,
                "final": state.get("final") or state.get("answer", ""),
                **_event("review", f"approved after round {review_round}"),
            }
        if review_round < max_reviews:
            critique: Critique = {
                "grounded": False,
                "issues": [feedback] if feedback else [],
                "unsupported_claims": [],
            }
            return {
                "approved": False,
                "review_round": review_round,
                "critique": critique,
                **_event("review", f"rejected after round {review_round}; rewriting"),
            }
        draft = state.get("final") or state.get("answer", "")
        feedback_note = feedback or "no feedback given"
        annotated = f"{draft}\n\n[Review notice: draft rejected by reviewer - {feedback_note}]"
        return {
            "approved": False,
            "review_round": review_round,
            "final": annotated,
            "notice": "Draft rejected by reviewer; review budget exhausted.",
            **_event("review", f"rejected after round {review_round}; review budget exhausted"),
        }

    def route_after_grade(state: LoopRAGState) -> str:
        if state.get("relevant"):
            return "generate"
        if int(state.get("attempt", 0)) < max_attempts:
            return "rewrite"
        return "insufficient"

    def route_after_verify(state: LoopRAGState) -> str:
        if state.get("grounded"):
            return "finalize"
        if int(state.get("attempt", 0)) < max_attempts:
            return "rewrite"
        return "finalize"

    def route_after_review(state: LoopRAGState) -> str:
        if state.get("approved"):
            return "end"
        if int(state.get("review_round", 0)) < max_reviews:
            return "rewrite"
        return "end"

    graph = StateGraph(LoopRAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_node("verify", verify)
    graph.add_node("insufficient", insufficient)
    graph.add_node("finalize", finalize)
    if enable_review:
        graph.add_node("review", review)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "insufficient": "insufficient"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "verify")
    graph.add_conditional_edges(
        "verify", route_after_verify, {"finalize": "finalize", "rewrite": "rewrite"}
    )
    graph.add_edge("insufficient", END)
    if enable_review:
        graph.add_edge("finalize", "review")
        graph.add_conditional_edges(
            "review", route_after_review, {"end": END, "rewrite": "rewrite"}
        )
    else:
        graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


def run_config(thread_id: str, recursion_limit: int = 60) -> dict:
    """Invocation config shared by the CLI and tests."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": recursion_limit,
    }
