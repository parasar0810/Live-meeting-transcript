"""
Pydantic models for all WebSocket message types.

Design principle: every message type has an explicit `type` discriminator
field so the client can `switch(msg.type)` without ambiguity.
"""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class AgentType(str, Enum):
    FACT_CHECK = "fact_check"
    ACTION_ITEM = "action_item"
    CONTEXT_RAG = "context_rag"
    ROLLING_SUMMARY = "rolling_summary"


class FactCheckVerdict(str, Enum):
    VERIFIED = "verified"
    CONTRADICTION = "contradiction"
    UNVERIFIABLE = "unverifiable"


# ---------------------------------------------------------------------------
# Inbound (ingestor → server)
# ---------------------------------------------------------------------------


class TranscriptChunk(BaseModel):
    """A single speaker utterance delivered by the transcript simulator."""

    session_id: str = Field(..., min_length=1, max_length=128)
    chunk_id: int = Field(..., ge=0, description="Monotonically increasing, 0-indexed")
    speaker: str = Field(..., min_length=1, max_length=64)
    text: str = Field(..., min_length=1)
    timestamp: float = Field(default_factory=time.time)

    @model_validator(mode="after")
    def text_not_whitespace_only(self) -> "TranscriptChunk":
        if not self.text.strip():
            raise ValueError("text must contain non-whitespace content")
        return self


class MeetingEndSignal(BaseModel):
    """Sent by simulator to mark the end of the transcript feed."""

    type: Literal["meeting_end"] = "meeting_end"
    session_id: str


# ---------------------------------------------------------------------------
# Agent outputs (server → clients)
# ---------------------------------------------------------------------------


class Citation(BaseModel):
    """A reference to a retrieved knowledge-base document."""

    source: str = Field(..., description="Document filename or title")
    excerpt: str = Field(..., description="Verbatim excerpt used in reasoning")


class FactCheckResult(BaseModel):
    claim: str
    verdict: FactCheckVerdict
    explanation: str
    citations: list[Citation] = Field(default_factory=list)


class ActionItem(BaseModel):
    assignee: str
    task: str
    deadline: Optional[str] = None


class ContextResolution(BaseModel):
    reference: str = Field(..., description="The ambiguous pronoun or term")
    resolved_to: str = Field(..., description="What it refers to")
    explanation: str
    citations: list[Citation] = Field(default_factory=list)


class AgentOutput(BaseModel):
    """Wrapper sent to all WebSocket clients when an agent produces a result."""

    type: Literal["agent_output"] = "agent_output"
    agent_type: AgentType
    session_id: str
    chunk_id: int = Field(..., description="The chunk that triggered this agent")
    payload: Any = Field(
        ...,
        description=(
            "FactCheckResult | list[ActionItem] | list[ContextResolution] | str "
            "depending on agent_type"
        ),
    )
    timestamp: float = Field(default_factory=time.time)


# ---------------------------------------------------------------------------
# Session/infrastructure messages (server → clients)
# ---------------------------------------------------------------------------


class ChunkAck(BaseModel):
    """Acknowledgement sent to the ingestor after a chunk is processed."""

    type: Literal["chunk_ack"] = "chunk_ack"
    chunk_id: int
    agents_triggered: list[AgentType]


class ErrorMessage(BaseModel):
    """Non-fatal error returned to the ingestor or client."""

    type: Literal["error"] = "error"
    code: str
    detail: str
    chunk_id: Optional[int] = None


class CatchUpPayload(BaseModel):
    """Full session history sent to a late-joining client on connect."""

    type: Literal["catchup"] = "catchup"
    session_id: str
    transcript: list[TranscriptChunk]
    agent_outputs: list[AgentOutput]
    summary: Optional[str] = None


class ChunkBroadcast(BaseModel):
    """Broadcast a new transcript chunk to all connected clients."""

    type: Literal["transcript_chunk"] = "transcript_chunk"
    session_id: str
    chunk_id: int
    speaker: str
    text: str
    timestamp: float


class MeetingEndBroadcast(BaseModel):
    """Broadcast to all clients when the simulator signals meeting end."""

    type: Literal["meeting_end"] = "meeting_end"
    session_id: str


# ---------------------------------------------------------------------------
# Union type for outbound messages (used in type annotations)
# ---------------------------------------------------------------------------

OutboundMessage = Union[
    AgentOutput,
    ChunkBroadcast,
    ChunkAck,
    ErrorMessage,
    CatchUpPayload,
    MeetingEndBroadcast,
]
