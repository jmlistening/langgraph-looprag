"""OpenAI-compatible policy using explicit JSON prompts.

No provider-specific structured-output APIs are used: every decision is a
plain chat request whose text is parsed defensively. Malformed grader or
verifier output falls back conservatively rather than crashing the graph.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .policy import PolicyError, content_terms, token_coverage, truncate
from .state import Chunk, Critique, Grade

GRADE_SYSTEM = (
    "You are an evidence grader. Decide whether the supplied chunk can help "
    "answer the question. Reply with only JSON: "
    '{"relevant": true|false, "reason": "short reason"}. No prose, no code fences.'
)

REWRITE_SYSTEM = (
    "You rewrite search queries for a BM25 retriever. Reply with only the improved "
    "query, one line, at most 200 characters, no quotes, no explanation."
)

GENERATE_SYSTEM = (
    "You answer strictly from numbered sources. Cite every factual claim with [n] "
    "using the source number. Use only the supplied evidence. Answer in the "
    "question's language. If the evidence is insufficient, say so explicitly "
    "instead of guessing."
)

VERIFY_SYSTEM = (
    "You verify answers against sources. Every factual claim must be supported by "
    "the provided sources, and a missing citation for a substantive claim is an "
    "issue. Reply with only JSON: "
    '{"grounded": true|false, "issues": ["..."], "unsupported_claims": ["..."]}. '
    "No prose, no code fences."
)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return str(content)


def _parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    payload = json.loads(cleaned[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("JSON payload is not an object")
    return payload


class LLMPolicy:
    """A chat-model policy for OpenAI and OpenAI-compatible endpoints."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        *,
        temperature: float = 0.0,
    ) -> None:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise PolicyError(
                "Online mode needs the optional 'openai' dependency. "
                "Install it with: uv sync --extra openai (or pip install 'looprag[openai]'), "
                "or retry with --offline."
            ) from exc
        if not api_key:
            raise PolicyError(
                "Online mode requires LOOPRAG_API_KEY or OPENAI_API_KEY; retry with --offline."
            )
        self._model = model
        self.name = f"llm:{model}"
        try:
            self._llm = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
            )
        except Exception as exc:  # pragma: no cover - configuration errors
            raise PolicyError(
                f"Could not initialise the chat model '{model}'. "
                "Check LOOPRAG_BASE_URL, or retry with --offline."
            ) from exc

    # --------------------------------------------------------------- plumbing
    def _invoke(self, system: str, human: str) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        try:
            response = self._llm.invoke(
                [SystemMessage(content=system), HumanMessage(content=human)]
            )
        except Exception as exc:
            raise PolicyError(
                f"Chat completion failed for model '{self._model}'. "
                "Check your endpoint and credentials, or retry with --offline."
            ) from exc
        return _content_to_text(response.content)  # type: ignore[union-attr]

    @staticmethod
    def _sources_block(docs: list[Chunk], limit: int = 1500) -> str:
        blocks: list[str] = []
        for number, doc in enumerate(docs, start=1):
            blocks.append(
                f"[{number}] {doc.get('source', '?')} - {doc.get('title', 'Untitled')}\n"
                f"{truncate(doc.get('content', ''), limit)}"
            )
        return "\n\n".join(blocks)

    # ------------------------------------------------------------------ grade
    def grade(self, question: str, docs: list[Chunk]) -> list[Grade]:
        grades: list[Grade] = []
        for doc in docs:
            human = f"Question:\n{question}\n\nChunk:\n{truncate(doc.get('content', ''), 1200)}"
            try:
                payload = _parse_json_object(self._invoke(GRADE_SYSTEM, human))
                relevant = payload.get("relevant")
                if not isinstance(relevant, bool):
                    raise ValueError("relevant is not a boolean")
                reason = payload.get("reason")
                reason = reason.strip() if isinstance(reason, str) else ""
                grades.append({"relevant": relevant, "reason": reason or "no reason given"})
            except (ValueError, KeyError, TypeError):
                coverage = token_coverage(
                    content_terms(question),
                    f"{doc.get('title', '')}\n{doc.get('content', '')}",
                )
                grades.append(
                    {
                        "relevant": coverage >= 0.5,
                        "reason": (
                            f"fallback: malformed grader response; lexical coverage {coverage:.2f}"
                        ),
                    }
                )
        return grades

    # ---------------------------------------------------------------- rewrite
    def rewrite(self, question: str, query: str, feedback: list[str], attempt: int) -> str:
        human = (
            f"Original question:\n{question}\n\n"
            f"Current query:\n{query}\n\n"
            f"Attempt number: {attempt}\n\n"
            f"Feedback:\n" + ("\n".join(f"- {item}" for item in feedback) or "- none")
        )
        try:
            text = self._invoke(REWRITE_SYSTEM, human).strip()
        except PolicyError:
            raise
        text = text.splitlines()[0].strip() if text.splitlines() else ""
        text = text.strip("\"'` ").strip()
        return text[:200] if text else query

    # --------------------------------------------------------------- generate
    def generate(self, question: str, docs: list[Chunk], critique: Critique | None) -> str:
        if not docs:
            return "I don't have enough evidence in the corpus to answer that."
        human = f"Question:\n{question}\n\nSources:\n{self._sources_block(docs)}"
        if critique and critique.get("issues"):
            human += "\n\nIssues from a previous draft to avoid:\n" + "\n".join(
                f"- {issue}" for issue in critique["issues"]
            )
        text = self._invoke(GENERATE_SYSTEM, human).strip()
        return text or "I don't have enough evidence in the corpus to answer that."

    # ----------------------------------------------------------------- verify
    def verify(self, question: str, answer: str, docs: list[Chunk]) -> Critique:
        human = (
            f"Question:\n{question}\n\nAnswer:\n{answer}\n\nSources:\n{self._sources_block(docs)}"
        )
        try:
            payload = _parse_json_object(self._invoke(VERIFY_SYSTEM, human))
            grounded = payload.get("grounded")
            if not isinstance(grounded, bool):
                raise ValueError("grounded is not a boolean")
            issues = payload.get("issues", [])
            unsupported = payload.get("unsupported_claims", [])
            issues = [str(item) for item in issues] if isinstance(issues, list) else []
            unsupported = (
                [str(item) for item in unsupported] if isinstance(unsupported, list) else []
            )
            return {
                "grounded": grounded,
                "issues": issues,
                "unsupported_claims": unsupported,
            }
        except (ValueError, KeyError, TypeError):
            return {
                "grounded": False,
                "issues": ["fallback: verifier response could not be parsed"],
                "unsupported_claims": [],
            }
