"""
Session state management and fan-out broadcast.

Core design:
- SessionState holds all per-session mutable state: transcript buffer,
  agent output log, connected client queues, and the bounded work queue
  used for backpressure.
- SessionRegistry is a module-level dict protected by an asyncio.Lock. It
  serializes session creation/deletion but NOT reads — reads happen in hot
  paths and must stay lock-free (Python dict reads are GIL-protected enough
  for our purposes).
- Fan-out: each connected client gets its own asyncio.Queue. broadcast()
  enqueues into every client queue without awaiting. If a client's queue is
  full (slow consumer), that client's update is dropped silently — we never
  block the broadcast path for one bad client.
- Backpressure on work_queue: see _enqueue_work() below.

Fan-out mechanism — why in-process queues, not Redis:
  This system runs as a single Uvicorn process (even with --workers=1, which
  is the documented setup). In-process asyncio.Queues give zero serialization
  overhead and sub-millisecond delivery. Redis pub/sub would be required if
  we ran N>1 server processes (e.g., Gunicorn with multiple workers) because
  client WebSocket connections would be distributed across processes and a
  broadcast from process A would not reach clients on process B. That
  threshold has NOT been crossed here. DESIGN.md calls this out explicitly
  with the exact failure mode.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from backend.config import settings
from backend.models import AgentOutput, CatchUpPayload, OutboundMessage, TranscriptChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class SessionState:
    """All mutable state for one active session."""

    session_id: str

    # Ordered transcript buffer — append-only at runtime.
    transcript: list[TranscriptChunk] = field(default_factory=list)

    # Log of all agent outputs produced so far — used for late-joiner catch-up.
    agent_output_log: list[AgentOutput] = field(default_factory=list)

    # Current rolling summary text (updated by Rolling Summary agent in place).
    current_summary: Optional[str] = None

    # Connected browser-client queues.  Each client gets its own queue.
    client_queues: set[asyncio.Queue] = field(default_factory=set)

    # Bounded work queue consumed by the agent worker coroutine.
    # maxsize enforces backpressure — see _enqueue_work().
    work_queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=settings.work_queue_maxsize)
    )

    # Chunk ordering guard.
    last_chunk_id: int = -1

    # Tracks when the Rolling Summary agent last ran (for time-based trigger).
    last_summary_chunk_id: int = -1
    last_summary_time: float = field(default_factory=time.time)

    # True once the meeting_end signal is received.
    meeting_ended: bool = False


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------


class SessionRegistry:
    """
    Singleton registry of all active sessions.

    Thread-safety note: asyncio is single-threaded; all mutations happen on
    the event loop. The lock below is an asyncio.Lock (not threading.Lock).
    It guards session creation only — once a session exists, its internal
    state is mutated without the registry lock.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(self, session_id: str) -> SessionState:
        """Return existing session or create a new one."""
        if session_id in self._sessions:
            return self._sessions[session_id]
        async with self._lock:
            # Double-check under lock.
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState(session_id=session_id)
                logger.info("Session created: %s", session_id)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[SessionState]:
        """Return session state or None if not found."""
        return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> None:
        async with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info("Session deleted: %s", session_id)

    def all_sessions(self) -> dict[str, SessionState]:
        return dict(self._sessions)


# Module-level singleton
registry = SessionRegistry()


# ---------------------------------------------------------------------------
# Client connection management
# ---------------------------------------------------------------------------


def add_client(session: SessionState) -> asyncio.Queue:
    """
    Register a new browser client for this session.

    Returns a fresh asyncio.Queue the WebSocket handler should drain.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=settings.client_queue_maxsize)
    session.client_queues.add(q)
    logger.debug(
        "Client added to session %s (total clients: %d)",
        session.session_id,
        len(session.client_queues),
    )
    return q


def remove_client(session: SessionState, q: asyncio.Queue) -> None:
    """Deregister a disconnected client."""
    session.client_queues.discard(q)
    logger.debug(
        "Client removed from session %s (remaining: %d)",
        session.session_id,
        len(session.client_queues),
    )


# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------


def broadcast(session: SessionState, message: OutboundMessage) -> None:
    """
    Push `message` into every connected client's queue.

    This is a synchronous call (no await) so it can be called from within
    agent threads via loop.call_soon_threadsafe if needed. The per-client
    queue is bounded; put_nowait() raises asyncio.QueueFull if the client
    is too slow. We catch that and drop rather than blocking.
    """
    payload = message.model_dump_json()
    dropped = 0
    for q in list(session.client_queues):  # snapshot to avoid mutation during iteration
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dropped += 1

    if dropped:
        logger.warning(
            "Dropped update for %d slow client(s) in session %s",
            dropped,
            session.session_id,
        )


# ---------------------------------------------------------------------------
# Catch-up payload (for late joiners)
# ---------------------------------------------------------------------------


def build_catchup(session: SessionState) -> CatchUpPayload:
    """
    Build the full session history message sent to a client that connects
    after the meeting has already started.

    Late-joiner policy (from DESIGN.md):
        Send full history — transcript + all prior agent outputs. The
        alternative (new-updates-only) was rejected because a late joiner
        seeing action items and fact-checks with no transcript context would
        have a confusing and incomplete UX. The payload can be large for a
        long meeting, but a WebSocket send is a single write; there is no
        reason to page it.
    """
    return CatchUpPayload(
        session_id=session.session_id,
        transcript=list(session.transcript),
        agent_outputs=list(session.agent_output_log),
        summary=session.current_summary,
    )


# ---------------------------------------------------------------------------
# Work queue (backpressure)
# ---------------------------------------------------------------------------


async def enqueue_work(session: SessionState, task: object) -> bool:
    """
    Attempt to place an agent task on the session's bounded work queue.

    Backpressure policy (from DESIGN.md):
        If the queue is full, evict the *oldest* pending item before inserting
        the new one. Rationale: the most recent chunk is the most actionable
        context; losing an older pending trigger is less harmful. We never
        block the ingestion WebSocket — that would stall the transcript feed.

    Returns True if the task was enqueued (after possible eviction),
            False only if an unexpected error occurred.
    """
    q = session.work_queue
    if q.full():
        try:
            evicted = q.get_nowait()
            logger.warning(
                "Work queue full for session %s — evicted oldest task %r to make room",
                session.session_id,
                evicted,
            )
        except asyncio.QueueEmpty:
            pass  # Race between full-check and get; safe to ignore.
    try:
        q.put_nowait(task)
        return True
    except asyncio.QueueFull:
        logger.error(
            "Could not enqueue work for session %s even after eviction",
            session.session_id,
        )
        return False
