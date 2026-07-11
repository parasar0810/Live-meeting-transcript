"""
FastAPI application — WebSocket ingestion and client fan-out.

Endpoints:
    POST /sessions                         — create a session
    WS   /ws/{session_id}/ingest           — simulator feeds chunks here
    WS   /ws/{session_id}/client           — browser clients connect here
    GET  /sessions/{session_id}/state      — debug: current session state
    GET  /health                           — health check

Concurrency architecture:
    - The ingestion handler is async; it validates the chunk, appends to the
      transcript buffer, runs the supervisor (sync/fast), and enqueues
      AgentTasks onto the session's work_queue — all without awaiting any LLM.
    - A per-session worker coroutine (`_run_agent_worker`) runs in the
      background for the lifetime of the session. It drains the work queue
      and dispatches each batch of tasks to `run_tasks_concurrent`, which
      uses asyncio.gather so all triggered agents for a chunk start together.
    - When an agent completes, it calls broadcast() to push the result to
      every connected client queue. The client WebSocket handler drains its
      own queue and sends the JSON to the browser.

This architecture means:
    - New chunks are never blocked by in-flight agents.
    - Slow agents don't block fast agents.
    - All connected clients receive the same message stream.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from backend.agents import base as agent_base
# Import agents so their @register_agent decorators fire at startup.
import backend.agents.fact_check        # noqa: F401
import backend.agents.action_item       # noqa: F401
import backend.agents.context_rag       # noqa: F401
import backend.agents.rolling_summary   # noqa: F401

from backend.config import settings
from backend.models import (
    AgentOutput,
    AgentType,
    ChunkAck,
    ChunkBroadcast,
    ErrorMessage,
    MeetingEndSignal,
    TranscriptChunk,
)
from backend.rag.chroma_store import get_chroma_store
from backend.session_manager import (
    add_client,
    broadcast,
    build_catchup,
    enqueue_work,
    registry,
    remove_client,
)
from backend.supervisor import AgentTask, build_tasks

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan: warm up ChromaDB and thread pool at startup
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — initialising ChromaDB…")
    get_chroma_store()  # Eager init; logs if collection is empty.
    agent_base.get_executor()  # Eager init of thread pool.
    logger.info("Startup complete.")
    yield
    logger.info("Shutting down — closing thread pool…")
    agent_base.shutdown_executor()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Live Meeting Intelligence Copilot",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Per-session background worker tasks (asyncio.Task handles).
_session_workers: dict[str, asyncio.Task] = {}


# ---------------------------------------------------------------------------
# Per-session agent worker
# ---------------------------------------------------------------------------


async def _run_agent_worker(session_id: str) -> None:
    """
    Background coroutine: drains the session work_queue and runs agents.

    Each item dequeued is an AgentTask. After running it, the result is:
      1. Stored in session.agent_output_log (for late-joiner catch-up).
      2. Broadcast to all connected clients.
      3. If it's a Rolling Summary, session.current_summary is updated.
    """
    from backend.agents.base import run_tasks_concurrent

    logger.info("Agent worker started for session %s", session_id)
    session = await registry.get_or_create(session_id)

    while True:
        try:
            task: AgentTask = await asyncio.wait_for(
                session.work_queue.get(), timeout=60.0
            )
        except asyncio.TimeoutError:
            # No work for 60s — check if session still exists.
            if registry.get(session_id) is None:
                logger.info("Agent worker exiting: session %s gone", session_id)
                return
            continue
        except asyncio.CancelledError:
            logger.info("Agent worker cancelled for session %s", session_id)
            return

        # Gather all tasks for this "batch" — items enqueued for the same
        # chunk may be in rapid succession; we process one task at a time
        # but dispatch them concurrently if multiple arrive at once.
        # We collect any immediately available tasks for the same chunk to
        # batch them into one asyncio.gather call.
        batch = [task]
        same_chunk_id = task.trigger_chunk.chunk_id
        while True:
            try:
                next_task = session.work_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if next_task.trigger_chunk.chunk_id == same_chunk_id:
                batch.append(next_task)
            else:
                # Different chunk — re-queue and stop batching.
                await enqueue_work(session, next_task)
                break

        outputs = await run_tasks_concurrent(batch)

        for output in outputs:
            # Persist to session log for catch-up.
            session.agent_output_log.append(output)

            # Update rolling summary in-place.
            if output.agent_type == AgentType.ROLLING_SUMMARY:
                session.current_summary = output.payload
                session.last_summary_chunk_id = output.chunk_id
                session.last_summary_time = time.time()

            # Fan-out to all connected clients.
            broadcast(session, output)
            logger.debug(
                "Broadcast %s output (chunk %d) to %d client(s)",
                output.agent_type.value,
                output.chunk_id,
                len(session.client_queues),
            )

        for _ in batch:
            session.work_queue.task_done()


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "sessions": len(registry.all_sessions())}


@app.post("/sessions")
async def create_session():
    session_id = str(uuid.uuid4())[:8]
    await registry.get_or_create(session_id)
    _ensure_worker(session_id)
    logger.info("Session created via REST: %s", session_id)
    return {"session_id": session_id}


@app.get("/sessions/{session_id}/state")
async def get_session_state(session_id: str):
    session = registry.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session_id,
        "chunk_count": len(session.transcript),
        "last_chunk_id": session.last_chunk_id,
        "connected_clients": len(session.client_queues),
        "agent_output_count": len(session.agent_output_log),
        "current_summary": session.current_summary,
        "meeting_ended": session.meeting_ended,
        "work_queue_size": session.work_queue.qsize(),
    }


def _ensure_worker(session_id: str) -> None:
    """Start a background worker for the session if one isn't running."""
    task = _session_workers.get(session_id)
    if task is None or task.done():
        worker = asyncio.create_task(
            _run_agent_worker(session_id),
            name=f"agent-worker-{session_id}",
        )
        _session_workers[session_id] = worker
        logger.info("Background agent worker started for session %s", session_id)


# ---------------------------------------------------------------------------
# WebSocket: ingestion endpoint (simulator → server)
# ---------------------------------------------------------------------------


@app.websocket("/ws/{session_id}/ingest")
async def ingest_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info("Ingestor connected for session %s", session_id)

    session = await registry.get_or_create(session_id)
    _ensure_worker(session_id)

    try:
        while True:
            raw = await websocket.receive_text()

            # --- Parse message ---
            try:
                data = json.loads(raw)
            except json.JSONDecodeError as exc:
                await websocket.send_text(
                    ErrorMessage(code="invalid_json", detail=str(exc)).model_dump_json()
                )
                continue

            # Handle meeting_end signal
            if data.get("type") == "meeting_end":
                session.meeting_ended = True
                logger.info("Meeting ended for session %s", session_id)
                from backend.models import MeetingEndBroadcast
                broadcast(session, MeetingEndBroadcast(session_id=session_id))
                await websocket.send_text(
                    json.dumps({"type": "ack", "detail": "meeting_end received"})
                )
                break

            # --- Validate chunk ---
            try:
                chunk = TranscriptChunk.model_validate({**data, "session_id": session_id})
            except ValidationError as exc:
                await websocket.send_text(
                    ErrorMessage(
                        code="validation_error",
                        detail=str(exc),
                        chunk_id=data.get("chunk_id"),
                    ).model_dump_json()
                )
                continue  # Session survives; next chunk is processed normally.

            # --- Order guard ---
            expected_id = session.last_chunk_id + 1
            if chunk.chunk_id != expected_id:
                if chunk.chunk_id <= session.last_chunk_id:
                    # Duplicate — silently discard.
                    logger.warning(
                        "Duplicate chunk %d (last: %d) for session %s — discarding",
                        chunk.chunk_id, session.last_chunk_id, session_id,
                    )
                    await websocket.send_text(
                        ErrorMessage(
                            code="duplicate_chunk",
                            detail=f"chunk_id {chunk.chunk_id} already received",
                            chunk_id=chunk.chunk_id,
                        ).model_dump_json()
                    )
                else:
                    # Gap — warn but still accept (guard, not full reorder buffer).
                    logger.warning(
                        "Out-of-order chunk: expected %d, got %d for session %s",
                        expected_id, chunk.chunk_id, session_id,
                    )
                    # Advance last_chunk_id to received to avoid deadlock.
                    session.last_chunk_id = chunk.chunk_id
                    session.transcript.append(chunk)
                    # Send a warning back but continue processing.
                    await websocket.send_text(
                        ErrorMessage(
                            code="out_of_order_chunk",
                            detail=f"Expected {expected_id}, got {chunk.chunk_id}. Accepted with warning.",
                            chunk_id=chunk.chunk_id,
                        ).model_dump_json()
                    )
                    # Still broadcast the chunk to clients.
                    broadcast(session, _build_chunk_broadcast(chunk))
                    _dispatch_agents(chunk, session)
                    continue
            else:
                session.last_chunk_id = chunk.chunk_id
                session.transcript.append(chunk)

            # --- Broadcast chunk to all clients ---
            broadcast(session, _build_chunk_broadcast(chunk))

            # --- Supervisor: decide which agents to trigger & ack ---
            triggered_types = _dispatch_agents(chunk, session)

            # --- Ack to ingestor ---
            await websocket.send_text(
                ChunkAck(
                    chunk_id=chunk.chunk_id,
                    agents_triggered=triggered_types,
                ).model_dump_json()
            )

    except WebSocketDisconnect:
        logger.info("Ingestor disconnected from session %s", session_id)
    except Exception as exc:
        logger.error("Ingest error in session %s: %s", session_id, exc, exc_info=True)
        try:
            await websocket.send_text(
                ErrorMessage(code="internal_error", detail=str(exc)).model_dump_json()
            )
        except Exception:
            pass


def _build_chunk_broadcast(chunk: TranscriptChunk) -> "ChunkBroadcast":
    """Wrap a raw chunk as a typed broadcast so clients can render it in the transcript panel."""
    return ChunkBroadcast(
        session_id=chunk.session_id,
        chunk_id=chunk.chunk_id,
        speaker=chunk.speaker,
        text=chunk.text,
        timestamp=chunk.timestamp,
    )


def _dispatch_agents(chunk: TranscriptChunk, session) -> list[AgentType]:
    """Build and enqueue AgentTasks for the given chunk. Returns triggered AgentTypes."""
    tasks = build_tasks(
        chunk,
        session.transcript,
        session.last_summary_chunk_id,
        session.last_summary_time,
    )

    # For Rolling Summary, attach the incremental context.
    for task in tasks:
        if task.agent_type == AgentType.ROLLING_SUMMARY:
            # New chunks since last summary run.
            last_id = session.last_summary_chunk_id
            new_chunks = [
                c for c in session.transcript
                if c.chunk_id > last_id
            ]
            task.new_chunks = new_chunks  # type: ignore[attr-defined]
            task.previous_summary = session.current_summary  # type: ignore[attr-defined]

    # Enqueue all tasks (bounded queue with drop-oldest on overflow).
    for task in tasks:
        asyncio.create_task(enqueue_work(session, task))

    return [t.agent_type for t in tasks]


# ---------------------------------------------------------------------------
# WebSocket: client endpoint (browser → server)
# ---------------------------------------------------------------------------


@app.websocket("/ws/{session_id}/client")
async def client_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    logger.info("Browser client connected for session %s", session_id)

    session = await registry.get_or_create(session_id)
    _ensure_worker(session_id)

    # Register client and send catch-up payload.
    client_queue = add_client(session)

    try:
        # Send catch-up payload FIRST (full history for late joiners).
        if session.transcript or session.agent_output_log:
            catchup = build_catchup(session)
            await websocket.send_text(catchup.model_dump_json())
            logger.info(
                "Sent catch-up to new client for session %s (%d chunks, %d outputs)",
                session_id,
                len(catchup.transcript),
                len(catchup.agent_outputs),
            )

        # Stream live updates from the client queue.
        while True:
            try:
                msg = await asyncio.wait_for(client_queue.get(), timeout=30.0)
                await websocket.send_text(msg)
                client_queue.task_done()
            except asyncio.TimeoutError:
                # Send a ping to detect dead connections.
                try:
                    await websocket.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break

    except WebSocketDisconnect:
        logger.info("Browser client disconnected from session %s", session_id)
    except Exception as exc:
        logger.error("Client error in session %s: %s", session_id, exc, exc_info=True)
    finally:
        remove_client(session, client_queue)
