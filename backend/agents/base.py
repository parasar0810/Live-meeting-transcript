"""
Agent base: shared dataclasses and the agent runner.

The runner is the core concurrency mechanism:
  - Each AgentTask is dispatched to run_in_executor (thread pool), so
    LLM I/O (which releases the GIL on socket wait) runs in a real OS
    thread without blocking the event loop.
  - Multiple tasks from the same chunk are gathered with asyncio.gather,
    which starts all of them before awaiting any. This means Fact-Check
    and Action-Item run concurrently, not sequentially.
  - The per-session work_queue is drained by a single coroutine per session,
    but that coroutine uses asyncio.gather so multiple agents per chunk
    still overlap.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Awaitable

from backend.models import AgentOutput, AgentType
from backend.supervisor import AgentTask

logger = logging.getLogger(__name__)

# Module-level thread pool shared across all sessions.
# Sized to allow multiple concurrent agent calls across all active sessions.
_EXECUTOR: ThreadPoolExecutor | None = None


def get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        from backend.config import settings
        _EXECUTOR = ThreadPoolExecutor(
            max_workers=settings.agent_thread_workers,
            thread_name_prefix="agent-worker",
        )
        logger.info("Agent thread pool initialised with %d workers", settings.agent_thread_workers)
    return _EXECUTOR


def shutdown_executor() -> None:
    global _EXECUTOR
    if _EXECUTOR:
        _EXECUTOR.shutdown(wait=False)
        _EXECUTOR = None


# ---------------------------------------------------------------------------
# Agent registry — maps AgentType → async run function
# ---------------------------------------------------------------------------

AgentRunFn = Callable[[AgentTask], Awaitable[AgentOutput]]
_AGENT_REGISTRY: dict[AgentType, AgentRunFn] = {}


def register_agent(agent_type: AgentType) -> Callable:
    """Decorator: register an async agent function in the registry."""
    def decorator(fn: AgentRunFn) -> AgentRunFn:
        _AGENT_REGISTRY[agent_type] = fn
        logger.debug("Registered agent: %s → %s", agent_type.value, fn.__name__)
        return fn
    return decorator


def get_agent(agent_type: AgentType) -> AgentRunFn:
    if agent_type not in _AGENT_REGISTRY:
        raise KeyError(f"No agent registered for type: {agent_type}")
    return _AGENT_REGISTRY[agent_type]


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------


async def run_agent_task(
    task: AgentTask,
    loop: asyncio.AbstractEventLoop | None = None,
) -> AgentOutput | None:
    """
    Run a single AgentTask, returning its output or None on failure.

    The agent function itself is async (it may await LangGraph coroutines
    internally), so we just await it directly. Blocking LLM calls within
    the agent are wrapped in asyncio.to_thread / run_in_executor inside
    the agent implementation.
    """
    agent_fn = get_agent(task.agent_type)
    start = time.time()
    try:
        output = await agent_fn(task)
        elapsed = time.time() - start
        logger.info(
            "Agent %s completed for chunk %d in %.2fs",
            task.agent_type.value,
            task.trigger_chunk.chunk_id,
            elapsed,
        )
        return output
    except Exception as exc:
        logger.error(
            "Agent %s failed for chunk %d: %s",
            task.agent_type.value,
            task.trigger_chunk.chunk_id,
            exc,
            exc_info=True,
        )
        return None


async def run_tasks_concurrent(tasks: list[AgentTask]) -> list[AgentOutput]:
    """
    Run multiple agent tasks concurrently using asyncio.gather.

    All tasks start executing before any result is awaited. This is the
    key property that prevents Agent 1 (slow retrieval) from delaying
    Agent 2's output — they are independent futures.
    """
    if not tasks:
        return []

    logger.info(
        "Running %d agent(s) concurrently for chunk %d: %s",
        len(tasks),
        tasks[0].trigger_chunk.chunk_id,
        [t.agent_type.value for t in tasks],
    )

    results = await asyncio.gather(
        *[run_agent_task(t) for t in tasks],
        return_exceptions=False,  # Exceptions are caught inside run_agent_task
    )

    return [r for r in results if r is not None]
