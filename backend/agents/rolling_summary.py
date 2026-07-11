"""
Rolling Summary Agent — LangGraph StateGraph implementation.

Graph topology:
    summarize → emit

Incremental strategy:
    The agent receives (previous_summary, new_chunks_since_last_run).
    The LLM produces an *updated* summary, not a full re-summarization
    of the entire transcript. This keeps token cost bounded regardless
    of meeting length: the prompt size is O(new_chunks) + O(summary),
    not O(all_chunks).

This is genuinely a rolling summary, not a final summary:
    - Runs every N chunks (default: 5) or every M seconds (default: 30)
    - The summary stored in SessionState is replaced (not appended) on
      each run — it represents the current state of understanding
    - Clients see summary updates multiple times during the meeting
"""

from __future__ import annotations

import asyncio
import logging
from typing import TypedDict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from backend.agents.base import register_agent
from backend.config import settings
from backend.models import AgentOutput, AgentType
from backend.supervisor import AgentTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class RollingSummaryState(TypedDict):
    task: AgentTask
    previous_summary: Optional[str]
    new_chunks_text: str
    updated_summary: Optional[str]
    output: AgentOutput | None


# ---------------------------------------------------------------------------
# Node: summarize
# ---------------------------------------------------------------------------


async def _summarize(state: RollingSummaryState) -> RollingSummaryState:
    task = state["task"]
    previous_summary = state["previous_summary"]
    new_chunks_text = state["new_chunks_text"]

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        temperature=0.2,  # Slight creativity for natural summary prose
    )

    if previous_summary:
        system = SystemMessage(content=(
            "You are a live meeting summarizer. You maintain a running "
            "summary of an ongoing meeting, updating it as new transcript "
            "chunks arrive. "
            "Given the current summary and new transcript lines, produce "
            "an updated summary that incorporates the new information. "
            "Preserve important decisions, action items, and key facts from "
            "the previous summary. The updated summary should be concise "
            "(3–6 sentences) and written in present/past tense. "
            "Return ONLY the updated summary text, no headers or labels."
        ))
        human = HumanMessage(content=(
            f"Current summary:\n{previous_summary}\n\n"
            f"New transcript lines:\n{new_chunks_text}"
        ))
    else:
        system = SystemMessage(content=(
            "You are a live meeting summarizer. Given the opening lines of "
            "a meeting transcript, produce a concise summary (2–4 sentences) "
            "of what has been discussed so far. "
            "Return ONLY the summary text, no headers or labels."
        ))
        human = HumanMessage(content=f"Transcript:\n{new_chunks_text}")

    response = await asyncio.to_thread(
        lambda: llm.invoke([system, human]).content
    )

    updated_summary = response.strip()
    logger.debug(
        "Rolling Summary: updated for chunk %d (prev: %s, new_chunks_len: %d)",
        task.trigger_chunk.chunk_id,
        "yes" if previous_summary else "no",
        len(new_chunks_text),
    )
    return {**state, "updated_summary": updated_summary}


# ---------------------------------------------------------------------------
# Node: emit
# ---------------------------------------------------------------------------


def _emit(state: RollingSummaryState) -> RollingSummaryState:
    task = state["task"]
    output = AgentOutput(
        agent_type=AgentType.ROLLING_SUMMARY,
        session_id=task.session_id,
        chunk_id=task.trigger_chunk.chunk_id,
        payload=state["updated_summary"] or "",
    )
    return {**state, "output": output}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    graph = StateGraph(RollingSummaryState)
    graph.add_node("summarize", _summarize)
    graph.add_node("emit", _emit)

    graph.set_entry_point("summarize")
    graph.add_edge("summarize", "emit")
    graph.add_edge("emit", END)

    return graph.compile()


_ROLLING_SUMMARY_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Registered entry point
# ---------------------------------------------------------------------------


@register_agent(AgentType.ROLLING_SUMMARY)
async def run_rolling_summary(task: AgentTask) -> AgentOutput:
    """
    The session_manager passes the previous_summary and new_chunks into
    the task via extra fields stored on the AgentTask. We access them via
    getattr with defaults so this works even if they're missing (first run).
    """
    previous_summary: Optional[str] = getattr(task, "previous_summary", None)
    new_chunks: list = getattr(task, "new_chunks", task.transcript_snapshot)

    new_chunks_text = "\n".join(
        f"{c.speaker}: {c.text}" for c in new_chunks
    )

    initial_state: RollingSummaryState = {
        "task": task,
        "previous_summary": previous_summary,
        "new_chunks_text": new_chunks_text,
        "updated_summary": None,
        "output": None,
    }

    final_state = await _ROLLING_SUMMARY_GRAPH.ainvoke(initial_state)
    output = final_state.get("output")
    if output is None:
        raise RuntimeError("Rolling Summary graph did not produce an output")
    return output
