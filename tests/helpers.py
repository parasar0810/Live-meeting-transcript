"""
Shared test helpers — WebSocket clients, chunk builders, message collectors.

These helpers connect to the LIVE running server (not mocked).
The server must be running at SERVER_URL before tests execute.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Optional

import websockets
from websockets.exceptions import ConnectionClosed

SERVER_HTTP = "http://localhost:8000"
SERVER_WS = "ws://localhost:8000"

# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


async def create_session_http() -> str:
    """Create a session via the REST endpoint, return session_id."""
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{SERVER_HTTP}/sessions")
        resp.raise_for_status()
        return resp.json()["session_id"]


async def get_session_state(session_id: str) -> dict:
    """Query the debug state endpoint."""
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SERVER_HTTP}/sessions/{session_id}/state")
        resp.raise_for_status()
        return resp.json()


async def check_health() -> dict:
    """Hit the health endpoint."""
    import httpx
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{SERVER_HTTP}/health")
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# WebSocket connection helpers
# ---------------------------------------------------------------------------


async def connect_ingest(session_id: str):
    """Open a WebSocket to the ingest endpoint."""
    url = f"{SERVER_WS}/ws/{session_id}/ingest"
    ws = await websockets.connect(url, close_timeout=5)
    return ws


async def connect_client(session_id: str):
    """Open a WebSocket to the client endpoint."""
    url = f"{SERVER_WS}/ws/{session_id}/client"
    ws = await websockets.connect(url, close_timeout=5)
    return ws


# ---------------------------------------------------------------------------
# Chunk building
# ---------------------------------------------------------------------------


def build_chunk(chunk_id: int, speaker: str, text: str) -> dict:
    """Build a transcript chunk dict (session_id added server-side)."""
    return {
        "chunk_id": chunk_id,
        "speaker": speaker,
        "text": text,
        "timestamp": time.time(),
    }


# The full 44-line meeting transcript parsed into chunks
MEETING_CHUNKS: list[dict] = []


def _load_meeting_chunks() -> list[dict]:
    """Parse the meeting_transcript.txt into chunk dicts."""
    from pathlib import Path
    transcript_path = Path(__file__).resolve().parent.parent / "transcript_samples" / "meeting_transcript.txt"
    chunks = []
    chunk_id = 0
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            speaker, _, text = line.partition(":")
            speaker = speaker.strip()
            text = text.strip()
            if not speaker or not text:
                continue
            chunks.append(build_chunk(chunk_id, speaker, text))
            chunk_id += 1
    return chunks


def get_meeting_chunks() -> list[dict]:
    """Return cached meeting transcript chunks."""
    global MEETING_CHUNKS
    if not MEETING_CHUNKS:
        MEETING_CHUNKS = _load_meeting_chunks()
    return MEETING_CHUNKS


# ---------------------------------------------------------------------------
# Sending & receiving
# ---------------------------------------------------------------------------


async def send_chunk(ws, chunk: dict) -> dict | None:
    """Send a chunk and wait for the ack. Returns parsed ack or None on timeout."""
    await ws.send(json.dumps(chunk))
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
        return json.loads(raw)
    except asyncio.TimeoutError:
        return None


async def send_chunk_no_wait(ws, chunk: dict) -> None:
    """Send a chunk without waiting for ack."""
    await ws.send(json.dumps(chunk))


async def collect_messages(ws, timeout: float = 30.0, max_messages: int = 500) -> list[dict]:
    """
    Collect all messages from a WebSocket until timeout or max_messages reached.
    Filters out ping messages.
    """
    messages = []
    deadline = time.time() + timeout
    while time.time() < deadline and len(messages) < max_messages:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
            msg = json.loads(raw)
            if msg.get("type") != "ping":
                messages.append(msg)
        except asyncio.TimeoutError:
            continue
        except ConnectionClosed:
            break
    return messages


async def collect_until_agent_outputs(
    ws, expected_count: int, timeout: float = 60.0
) -> list[dict]:
    """
    Collect messages until we've received `expected_count` agent_output messages,
    or timeout. Returns ALL collected messages (including non-agent ones).
    """
    messages = []
    agent_count = 0
    deadline = time.time() + timeout
    while time.time() < deadline and agent_count < expected_count:
        remaining = deadline - time.time()
        if remaining <= 0:
            break
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=min(remaining, 5.0))
            msg = json.loads(raw)
            if msg.get("type") == "ping":
                continue
            messages.append(msg)
            if msg.get("type") == "agent_output":
                agent_count += 1
        except asyncio.TimeoutError:
            continue
        except ConnectionClosed:
            break
    return messages


def filter_by_type(messages: list[dict], msg_type: str) -> list[dict]:
    """Filter messages by their 'type' field."""
    return [m for m in messages if m.get("type") == msg_type]


def filter_agent_outputs(messages: list[dict], agent_type: str | None = None) -> list[dict]:
    """Filter for agent_output messages, optionally by agent_type."""
    outputs = filter_by_type(messages, "agent_output")
    if agent_type:
        outputs = [o for o in outputs if o.get("agent_type") == agent_type]
    return outputs
