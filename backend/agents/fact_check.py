"""
Fact-Check Agent — LangGraph StateGraph implementation.

Graph topology:
    extract_claims → retrieve_docs → cross_reference → emit

Node responsibilities:
    extract_claims:   Parse the trigger chunk for verifiable factual claims.
    retrieve_docs:    For each claim, query ChromaDB for relevant KB chunks.
    cross_reference:  LLM prompt: given claim + retrieved docs, verdict + explanation.
    emit:             Assemble AgentOutput with citations.

Citation policy:
    Every output MUST include which document(s) were used. If no relevant
    document was found in the KB, the verdict is "unverifiable" and
    citations is empty. The agent NEVER returns silence — it always produces
    an output, even if that output is "no claim found."

"Unverifiable" vs silence policy (from DESIGN.md):
    A claim with no KB coverage gets verdict=UNVERIFIABLE. Returning nothing
    would be indistinguishable from the fact-check not running at all. Returning
    UNVERIFIABLE gives the user actionable signal: "we checked and couldn't
    confirm this." This is the correct behavior for a meeting copilot where
    false confidence is more harmful than acknowledged uncertainty.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TypedDict, Annotated, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from backend.agents.base import register_agent
from backend.config import settings
from backend.models import (
    AgentOutput,
    AgentType,
    Citation,
    FactCheckResult,
    FactCheckVerdict,
)
from backend.rag.chroma_store import get_chroma_store
from backend.supervisor import AgentTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class FactCheckState(TypedDict):
    task: AgentTask
    claims: list[str]                            # extracted from chunk
    retrieved: dict[str, list[Any]]              # claim → list[RetrievedDoc]
    results: list[FactCheckResult]               # one per claim
    output: AgentOutput | None


# ---------------------------------------------------------------------------
# Node: extract_claims
# ---------------------------------------------------------------------------


async def _extract_claims(state: FactCheckState) -> FactCheckState:
    """Use LLM to extract factual claims from the trigger chunk."""
    task = state["task"]
    chunk_text = task.trigger_chunk.text
    speaker = task.trigger_chunk.speaker

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        temperature=0,
    )

    system = SystemMessage(content=(
        "You are a claim extractor. Given a meeting transcript chunk, "
        "extract all verifiable factual claims — statements that assert "
        "specific numbers, dates, product capabilities, market positions, "
        "or competitor facts. "
        "Return a JSON array of strings, each being one claim. "
        "If there are no verifiable claims, return an empty array []. "
        "Return ONLY the JSON array, no other text."
    ))
    human = HumanMessage(content=f"Speaker: {speaker}\nChunk: {chunk_text}")

    response = await asyncio.to_thread(
        lambda: llm.invoke([system, human]).content
    )

    try:
        raw = response.strip()
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        claims = json.loads(raw)
        if not isinstance(claims, list):
            claims = []
    except json.JSONDecodeError:
        logger.warning("Fact-Check: failed to parse claims JSON: %r", response)
        claims = []

    logger.debug("Fact-Check: extracted %d claim(s) from chunk %d", len(claims), task.trigger_chunk.chunk_id)
    return {**state, "claims": claims}


# ---------------------------------------------------------------------------
# Node: retrieve_docs
# ---------------------------------------------------------------------------


async def _retrieve_docs(state: FactCheckState) -> FactCheckState:
    """For each claim, run similarity search against ChromaDB."""
    claims = state["claims"]
    if not claims:
        return {**state, "retrieved": {}}

    store = get_chroma_store()
    retrieved: dict[str, list] = {}

    # Run all retrievals concurrently (each is a blocking ChromaDB call).
    async def _retrieve_one(claim: str) -> tuple[str, list]:
        docs = await asyncio.to_thread(
            lambda: store.similarity_search(claim, k=3, max_distance=0.8)
        )
        return claim, docs

    pairs = await asyncio.gather(*[_retrieve_one(c) for c in claims])
    retrieved = dict(pairs)

    total = sum(len(v) for v in retrieved.values())
    logger.debug("Fact-Check: retrieved %d doc chunk(s) across %d claim(s)", total, len(claims))
    return {**state, "retrieved": retrieved}


# ---------------------------------------------------------------------------
# Node: cross_reference
# ---------------------------------------------------------------------------


async def _cross_reference(state: FactCheckState) -> FactCheckState:
    """LLM cross-references each claim against its retrieved documents."""
    claims = state["claims"]
    retrieved = state["retrieved"]

    if not claims:
        return {**state, "results": []}

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        temperature=0,
    )

    async def _verify_one(claim: str) -> FactCheckResult:
        docs = retrieved.get(claim, [])

        if not docs:
            return FactCheckResult(
                claim=claim,
                verdict=FactCheckVerdict.UNVERIFIABLE,
                explanation="No relevant documents found in the knowledge base to verify this claim.",
                citations=[],
            )

        docs_text = "\n\n".join(
            f"[Source: {d.source}]\n{d.content}" for d in docs
        )

        system = SystemMessage(content=(
            "You are a fact-checker for a meeting AI assistant. "
            "Given a claim made in a meeting and relevant documents from "
            "the company knowledge base, determine if the claim is: "
            "VERIFIED (supported by the documents), "
            "CONTRADICTION (directly contradicted by the documents), or "
            "UNVERIFIABLE (documents exist but don't clearly confirm or deny). "
            "\n\nReturn a JSON object with exactly these fields: "
            '{"verdict": "verified|contradiction|unverifiable", '
            '"explanation": "one sentence", '
            '"cited_sources": ["source_name_1", ...]}. '
            "cited_sources must only include sources from the provided documents. "
            "Return ONLY the JSON object."
        ))
        human = HumanMessage(content=(
            f"Claim: {claim}\n\nKnowledge Base Documents:\n{docs_text}"
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
            verdict = FactCheckVerdict(data.get("verdict", "unverifiable").lower())
            explanation = data.get("explanation", "")
            cited = data.get("cited_sources", [])
            # Build Citation objects — include excerpt from matching retrieved doc
            citations = []
            for src in cited:
                matching = next((d for d in docs if d.source == src), None)
                if matching:
                    citations.append(Citation(
                        source=src,
                        excerpt=matching.content[:200],
                    ))
                else:
                    citations.append(Citation(source=src, excerpt=""))
            return FactCheckResult(
                claim=claim,
                verdict=verdict,
                explanation=explanation,
                citations=citations,
            )
        except Exception as exc:
            logger.error("Fact-Check: cross_reference parse error: %s — response: %r", exc, response)
            return FactCheckResult(
                claim=claim,
                verdict=FactCheckVerdict.UNVERIFIABLE,
                explanation=f"Parse error during fact-check: {exc}",
                citations=[],
            )

    results = await asyncio.gather(*[_verify_one(c) for c in claims])
    return {**state, "results": list(results)}


# ---------------------------------------------------------------------------
# Node: emit
# ---------------------------------------------------------------------------


def _emit(state: FactCheckState) -> FactCheckState:
    """Assemble the final AgentOutput."""
    task = state["task"]
    results = state["results"]

    if not results:
        # No claims extracted — emit a "nothing to check" placeholder so the
        # supervisor knows the agent ran (useful for debugging trigger heuristics).
        results = [FactCheckResult(
            claim="[No verifiable claims detected in this chunk]",
            verdict=FactCheckVerdict.UNVERIFIABLE,
            explanation="The chunk did not contain extractable factual claims.",
            citations=[],
        )]

    output = AgentOutput(
        agent_type=AgentType.FACT_CHECK,
        session_id=task.session_id,
        chunk_id=task.trigger_chunk.chunk_id,
        payload=[r.model_dump() for r in results],
    )
    return {**state, "output": output}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    graph = StateGraph(FactCheckState)
    graph.add_node("extract_claims", _extract_claims)
    graph.add_node("retrieve_docs", _retrieve_docs)
    graph.add_node("cross_reference", _cross_reference)
    graph.add_node("emit", _emit)

    graph.set_entry_point("extract_claims")
    graph.add_edge("extract_claims", "retrieve_docs")
    graph.add_edge("retrieve_docs", "cross_reference")
    graph.add_edge("cross_reference", "emit")
    graph.add_edge("emit", END)

    return graph.compile()


_FACT_CHECK_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Registered entry point
# ---------------------------------------------------------------------------


@register_agent(AgentType.FACT_CHECK)
async def run_fact_check(task: AgentTask) -> AgentOutput:
    initial_state: FactCheckState = {
        "task": task,
        "claims": [],
        "retrieved": {},
        "results": [],
        "output": None,
    }
    final_state = await _FACT_CHECK_GRAPH.ainvoke(initial_state)
    output = final_state.get("output")
    if output is None:
        raise RuntimeError("Fact-Check graph did not produce an output")
    return output
