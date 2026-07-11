"""
Context / RAG Agent — LangGraph StateGraph implementation.

Graph topology:
    extract_references → retrieve_docs → resolve → emit

Purpose: resolve ambiguous references — pronouns, informal shorthand terms,
and "the X" patterns — against the knowledge base and prior meeting history.

Example:
    "The migration is 60% done" → resolves "the migration" to
    "Python 2 → Python 3 threat detection pipeline migration"
    citing meeting_history.txt.

Citation policy: same as Fact-Check — every resolution must cite its
source document. An unresolvable reference is emitted as-is with an
explanation that it couldn't be resolved, rather than being silently dropped.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TypedDict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from backend.agents.base import register_agent
from backend.config import settings
from backend.models import AgentOutput, AgentType, Citation, ContextResolution
from backend.rag.chroma_store import get_chroma_store
from backend.supervisor import AgentTask

logger = logging.getLogger(__name__)

_CONTEXT_WINDOW = 8  # Wider than Action-Item; context resolution needs more history


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class ContextRAGState(TypedDict):
    task: AgentTask
    references: list[str]               # ambiguous terms extracted from chunk
    retrieved: dict[str, list[Any]]     # reference → list[RetrievedDoc]
    resolutions: list[ContextResolution]
    output: AgentOutput | None


# ---------------------------------------------------------------------------
# Node: extract_references
# ---------------------------------------------------------------------------


async def _extract_references(state: ContextRAGState) -> ContextRAGState:
    task = state["task"]
    chunk = task.trigger_chunk
    snapshot = task.transcript_snapshot

    recent = snapshot[-_CONTEXT_WINDOW:] if len(snapshot) >= _CONTEXT_WINDOW else snapshot
    context_lines = "\n".join(f"{c.speaker}: {c.text}" for c in recent)
    if not any(c.chunk_id == chunk.chunk_id for c in recent):
        context_lines += f"\n{chunk.speaker}: {chunk.text}"

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        temperature=0,
    )

    system = SystemMessage(content=(
        "You are an ambiguity detector for meeting transcripts. "
        "Given a meeting excerpt, identify all ambiguous references: "
        "pronouns (it, this, that, they, them) whose antecedent is unclear, "
        "informal shorthand ('the platform', 'the project', 'the migration', "
        "'the integration', 'the old system', 'the new dashboard'), and "
        "references to prior discussions ('as mentioned', 'last time', etc.). "
        "Return a JSON array of the ambiguous terms/phrases exactly as they "
        "appear in the text. If nothing is ambiguous, return []. "
        "Return ONLY the JSON array."
    ))
    human = HumanMessage(content=f"Transcript excerpt:\n{context_lines}")

    response = await asyncio.to_thread(
        lambda: llm.invoke([system, human]).content
    )

    try:
        raw = response.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        references = json.loads(raw)
        if not isinstance(references, list):
            references = []
        references = [r for r in references if isinstance(r, str) and r.strip()]
    except Exception as exc:
        logger.warning("Context-RAG: failed to parse references JSON: %s", exc)
        references = []

    logger.debug("Context-RAG: found %d reference(s) in chunk %d", len(references), chunk.chunk_id)
    return {**state, "references": references}


# ---------------------------------------------------------------------------
# Node: retrieve_docs
# ---------------------------------------------------------------------------


async def _retrieve_docs(state: ContextRAGState) -> ContextRAGState:
    references = state["references"]
    if not references:
        return {**state, "retrieved": {}}

    store = get_chroma_store()

    async def _retrieve_one(ref: str) -> tuple[str, list]:
        docs = await asyncio.to_thread(
            lambda: store.similarity_search(ref, k=3, max_distance=0.9)
        )
        return ref, docs

    pairs = await asyncio.gather(*[_retrieve_one(r) for r in references])
    retrieved = dict(pairs)
    total = sum(len(v) for v in retrieved.values())
    logger.debug("Context-RAG: retrieved %d doc chunk(s)", total)
    return {**state, "retrieved": retrieved}


# ---------------------------------------------------------------------------
# Node: resolve
# ---------------------------------------------------------------------------


async def _resolve(state: ContextRAGState) -> ContextRAGState:
    references = state["references"]
    retrieved = state["retrieved"]
    task = state["task"]

    if not references:
        return {**state, "resolutions": []}

    snapshot = task.transcript_snapshot
    recent = snapshot[-_CONTEXT_WINDOW:] if len(snapshot) >= _CONTEXT_WINDOW else snapshot
    context_lines = "\n".join(f"{c.speaker}: {c.text}" for c in recent)

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        temperature=0,
    )

    async def _resolve_one(ref: str) -> ContextResolution:
        docs = retrieved.get(ref, [])
        docs_text = "\n\n".join(
            f"[Source: {d.source}]\n{d.content}" for d in docs
        ) if docs else "No relevant documents found."

        system = SystemMessage(content=(
            "You are a reference resolver for a meeting intelligence system. "
            "Given an ambiguous reference from a meeting transcript, "
            "the recent transcript context, and knowledge base documents, "
            "determine what the reference most likely refers to. "
            "Return a JSON object with: "
            '{"resolved_to": "what it refers to", '
            '"explanation": "brief explanation citing evidence", '
            '"cited_sources": ["source1", ...]}. '
            "If you cannot determine the reference, set resolved_to to "
            '"[unresolvable]" and explain why. '
            "Return ONLY the JSON object."
        ))
        human = HumanMessage(content=(
            f"Ambiguous reference: \"{ref}\"\n\n"
            f"Recent transcript:\n{context_lines}\n\n"
            f"Knowledge base documents:\n{docs_text}"
        ))

        response = await asyncio.to_thread(
            lambda: llm.invoke([system, human]).content
        )

        try:
            raw = response.strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data = json.loads(raw)
            cited = data.get("cited_sources", [])
            citations = []
            for src in cited:
                matching = next((d for d in docs if d.source == src), None)
                if matching:
                    citations.append(Citation(source=src, excerpt=matching.content[:200]))
                else:
                    citations.append(Citation(source=src, excerpt=""))

            return ContextResolution(
                reference=ref,
                resolved_to=data.get("resolved_to", "[unresolvable]"),
                explanation=data.get("explanation", ""),
                citations=citations,
            )
        except Exception as exc:
            logger.error("Context-RAG: resolve parse error: %s", exc)
            return ContextResolution(
                reference=ref,
                resolved_to="[parse error]",
                explanation=str(exc),
                citations=[],
            )

    resolutions = await asyncio.gather(*[_resolve_one(r) for r in references])
    return {**state, "resolutions": list(resolutions)}


# ---------------------------------------------------------------------------
# Node: emit
# ---------------------------------------------------------------------------


def _emit(state: ContextRAGState) -> ContextRAGState:
    task = state["task"]
    output = AgentOutput(
        agent_type=AgentType.CONTEXT_RAG,
        session_id=task.session_id,
        chunk_id=task.trigger_chunk.chunk_id,
        payload=[r.model_dump() for r in state["resolutions"]],
    )
    return {**state, "output": output}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    graph = StateGraph(ContextRAGState)
    graph.add_node("extract_references", _extract_references)
    graph.add_node("retrieve_docs", _retrieve_docs)
    graph.add_node("resolve", _resolve)
    graph.add_node("emit", _emit)

    graph.set_entry_point("extract_references")
    graph.add_edge("extract_references", "retrieve_docs")
    graph.add_edge("retrieve_docs", "resolve")
    graph.add_edge("resolve", "emit")
    graph.add_edge("emit", END)

    return graph.compile()


_CONTEXT_RAG_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Registered entry point
# ---------------------------------------------------------------------------


@register_agent(AgentType.CONTEXT_RAG)
async def run_context_rag(task: AgentTask) -> AgentOutput:
    initial_state: ContextRAGState = {
        "task": task,
        "references": [],
        "retrieved": {},
        "resolutions": [],
        "output": None,
    }
    final_state = await _CONTEXT_RAG_GRAPH.ainvoke(initial_state)
    output = final_state.get("output")
    if output is None:
        raise RuntimeError("Context-RAG graph did not produce an output")
    return output
