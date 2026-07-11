"""
Action-Item Agent — LangGraph StateGraph implementation.

Graph topology:
    extract_items → emit

The graph is simple (single extraction step) but still uses LangGraph's
StateGraph for architectural consistency — all four agents can be reasoned
about the same way when reviewing the codebase.

Context window: the agent receives the trigger chunk + the 5 most recent
chunks for speaker attribution context. This handles the common pattern
where a speaker is named in one chunk ("Sarah, can you...") but the
commitment appears in the next ("Sure, I'll do that by Friday").
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
from backend.models import ActionItem, AgentOutput, AgentType
from backend.supervisor import AgentTask

logger = logging.getLogger(__name__)

_CONTEXT_WINDOW = 5  # Number of preceding chunks to include for speaker context


# ---------------------------------------------------------------------------
# LangGraph State
# ---------------------------------------------------------------------------


class ActionItemState(TypedDict):
    task: AgentTask
    items: list[ActionItem]
    output: AgentOutput | None


# ---------------------------------------------------------------------------
# Node: extract_items
# ---------------------------------------------------------------------------


async def _extract_items(state: ActionItemState) -> ActionItemState:
    task = state["task"]
    chunk = task.trigger_chunk
    snapshot = task.transcript_snapshot

    # Build a context window: last N chunks + trigger chunk
    recent = snapshot[-_CONTEXT_WINDOW:] if len(snapshot) >= _CONTEXT_WINDOW else snapshot
    context_lines = "\n".join(
        f"{c.speaker}: {c.text}" for c in recent
    )
    # Ensure the trigger chunk is included even if already in recent
    if not any(c.chunk_id == chunk.chunk_id for c in recent):
        context_lines += f"\n{chunk.speaker}: {chunk.text}"

    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_api_base,
        temperature=0,
    )

    system = SystemMessage(content=(
        "You are an action-item extractor for meeting notes. "
        "Given a meeting transcript excerpt, extract all action items — "
        "explicit commitments, tasks assigned to specific people, or "
        "items with deadlines. "
        "For each action item, identify: the assignee (person responsible), "
        "the task (what they will do), and the deadline (if mentioned, "
        "otherwise null). "
        "Return a JSON array of objects with fields: "
        '{"assignee": string, "task": string, "deadline": string|null}. '
        "If no action items are present, return []. "
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
        data = json.loads(raw)
        items = [
            ActionItem(
                assignee=d.get("assignee", "Unknown"),
                task=d.get("task", ""),
                deadline=d.get("deadline"),
            )
            for d in data
            if isinstance(d, dict) and d.get("task")
        ]
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("Action-Item: parse error: %s — response: %r", exc, response)
        items = []

    logger.debug(
        "Action-Item: extracted %d item(s) from chunk %d",
        len(items),
        chunk.chunk_id,
    )
    return {**state, "items": items}


# ---------------------------------------------------------------------------
# Node: emit
# ---------------------------------------------------------------------------


def _emit(state: ActionItemState) -> ActionItemState:
    task = state["task"]
    output = AgentOutput(
        agent_type=AgentType.ACTION_ITEM,
        session_id=task.session_id,
        chunk_id=task.trigger_chunk.chunk_id,
        payload=[item.model_dump() for item in state["items"]],
    )
    return {**state, "output": output}


# ---------------------------------------------------------------------------
# Build graph
# ---------------------------------------------------------------------------


def _build_graph() -> Any:
    graph = StateGraph(ActionItemState)
    graph.add_node("extract_items", _extract_items)
    graph.add_node("emit", _emit)

    graph.set_entry_point("extract_items")
    graph.add_edge("extract_items", "emit")
    graph.add_edge("emit", END)

    return graph.compile()


_ACTION_ITEM_GRAPH = _build_graph()


# ---------------------------------------------------------------------------
# Registered entry point
# ---------------------------------------------------------------------------


@register_agent(AgentType.ACTION_ITEM)
async def run_action_item(task: AgentTask) -> AgentOutput:
    initial_state: ActionItemState = {
        "task": task,
        "items": [],
        "output": None,
    }
    final_state = await _ACTION_ITEM_GRAPH.ainvoke(initial_state)
    output = final_state.get("output")
    if output is None:
        raise RuntimeError("Action-Item graph did not produce an output")
    return output
