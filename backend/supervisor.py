"""
Supervisor: heuristic-based agent trigger decision.

Design philosophy:
    The supervisor makes triggering decisions using compiled regex patterns
    evaluated against each incoming transcript chunk. No LLM call is made
    here — that would defeat the purpose of cost control.

    Each agent has its own trigger rule. The rules are intentionally
    conservative (high precision, moderate recall) because:
    1. False positives (triggering an agent unnecessarily) waste tokens.
    2. False negatives (missing a trigger) are tolerable — the system is a
       copilot, not a compliance recorder. Missed items can be caught in
       post-meeting review.

    Documented false-negative risks per agent — see DESIGN.md for full
    discussion:

    Action-Item agent:
        Misses commitments phrased as questions ("Could Sarah maybe handle
        that?") or as passive voice ("It was decided that..."). Also misses
        implicit commitments with no explicit subject ("Let's get that done
        by Friday" when the assignee is implied by conversation context).

    Fact-Check agent:
        Misses factual assertions made as rhetorical questions ("Aren't we
        already at 100k connections?") and claims embedded in subordinate
        clauses without a strong numerical/superlative marker.

    Context/RAG agent:
        Misses references that are highly domain-specific abbreviations not
        in the trigger list, and references that span multiple chunks (where
        the pronoun and its antecedent are in different chunks — the
        supervisor only sees the current chunk).

    Rolling Summary agent:
        Never misses — it fires on a time/chunk-count schedule regardless
        of content, not on content patterns.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

from backend.config import settings
from backend.models import AgentType, TranscriptChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compiled trigger patterns (compiled once at import time, O(1) per chunk)
# ---------------------------------------------------------------------------

# Action-Item triggers: commitment verbs, assignment language, deadline markers
_ACTION_ITEM_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bI['']ll\b", re.IGNORECASE),
    re.compile(r"\bI will\b", re.IGNORECASE),
    re.compile(r"\bwe will\b", re.IGNORECASE),
    re.compile(r"\bwe['']ll\b", re.IGNORECASE),
    re.compile(r"\bgoing to\b", re.IGNORECASE),
    re.compile(r"\bwill (take|handle|own|send|schedule|set up|write|build|fix|review|update|deploy|check|follow up)\b", re.IGNORECASE),
    re.compile(r"\b(action item|action:|todo:|task:)\b", re.IGNORECASE),
    re.compile(r"\b(by (monday|tuesday|wednesday|thursday|friday|saturday|sunday|end of (day|week|month|sprint|quarter)))\b", re.IGNORECASE),
    re.compile(r"\b(by (next week|eod|eow|cob))\b", re.IGNORECASE),
    re.compile(r"\b(assign(ed)? to|owner:|responsible for|point person)\b", re.IGNORECASE),
    re.compile(r"\bdeadline\b", re.IGNORECASE),
    re.compile(r"\bcan you (please |kindly )?(take|handle|own|send|schedule|write|build|fix|review)\b", re.IGNORECASE),
    re.compile(r"\bneed(s)? to\b", re.IGNORECASE),
]

# Fact-Check triggers: numeric claims, superlatives, dated facts, proper-noun assertions
_FACT_CHECK_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b\d{1,3}(,\d{3})*(\.\d+)?(%|k| million| billion| percent| connections| rows| users| dollars)\b", re.IGNORECASE),
    re.compile(r"\$\d+", re.IGNORECASE),
    re.compile(r"\b(fastest|slowest|largest|smallest|most|least|best|worst|only|first|last) (in|on|of|ever|among)\b", re.IGNORECASE),
    re.compile(r"\b(released|launched|shipped|announced) in (january|february|march|april|may|june|july|august|september|october|november|december|\d{4})\b", re.IGNORECASE),
    re.compile(r"\b(is|was|has|have|had|are|were) (the |a )?(fastest|biggest|only|first|last|top)\b", re.IGNORECASE),
    re.compile(r"\b(nova|atlas|guardian|pulse|streamcore|datavault|shieldai)\b.{0,40}\b(is|has|supports|processes|handles)\b", re.IGNORECASE),
    re.compile(r"\b\d{4}\b.{0,30}\b(released|launched|version|v\d)\b", re.IGNORECASE),
    re.compile(r"\bmarket share\b", re.IGNORECASE),
    re.compile(r"\b(soc2|iso 27001|hipaa|gdpr|certified|certification|compliant)\b", re.IGNORECASE),
    re.compile(r"\bP99\b|\bSLA\b|\buptime\b|\blatency\b|\bMTTD\b|\bMTTR\b", re.IGNORECASE),
]

# Context/RAG triggers: ambiguous pronouns after proper nouns, "the X" references,
# temporal references to prior discussions
_CONTEXT_RAG_PATTERNS: list[re.Pattern] = [
    # Pronoun following a product/team name in same or recent chunk
    re.compile(r"\b(nova|atlas|guardian|pulse|platform|phoenix|lighthouse|horizon|nexus).{0,60}\b(it|this|that|they|them|its)\b", re.IGNORECASE),
    re.compile(r"\b(it|this|that|they|them)\b.{0,40}\b(platform|project|migration|integration|system|feature|issue|problem|fix)\b", re.IGNORECASE),
    re.compile(r"\bthe (platform|project|migration|integration|system|old system|new dashboard|bridge)\b", re.IGNORECASE),
    re.compile(r"\b(last time|previously|as mentioned|as discussed|like we said|we talked about|you mentioned)\b", re.IGNORECASE),
    re.compile(r"\bthe (same|previous|prior|earlier) (issue|problem|approach|plan|decision|discussion|meeting)\b", re.IGNORECASE),
    re.compile(r"\bwhat (he|she|they|marcus|sarah|priya|jordan|felix|samara|amir|li) (said|meant|was talking about)\b", re.IGNORECASE),
    re.compile(r"\b(that work|that feature|that bug|that change|that ticket|that PR|that decision)\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Dataclass for a queued agent task
# ---------------------------------------------------------------------------


@dataclass
class AgentTask:
    """
    A unit of work dispatched to one agent.

    We snapshot the chunk and a shallow copy of the transcript buffer at
    dispatch time. This ensures the agent sees a consistent view of the
    transcript even if new chunks arrive while it is running.
    """
    agent_type: AgentType
    trigger_chunk: TranscriptChunk
    transcript_snapshot: list[TranscriptChunk]  # copy, not reference
    session_id: str
    enqueued_at: float = 0.0

    def __post_init__(self) -> None:
        if self.enqueued_at == 0.0:
            self.enqueued_at = time.time()

    def __repr__(self) -> str:
        return f"AgentTask({self.agent_type.value}, chunk={self.trigger_chunk.chunk_id})"


# ---------------------------------------------------------------------------
# Trigger decision functions
# ---------------------------------------------------------------------------


def _matches_any(text: str, patterns: list[re.Pattern]) -> bool:
    """Return True if `text` matches any pattern in `patterns`."""
    return any(p.search(text) for p in patterns)


def should_trigger_action_item(chunk: TranscriptChunk) -> bool:
    return _matches_any(chunk.text, _ACTION_ITEM_PATTERNS)


def should_trigger_fact_check(chunk: TranscriptChunk) -> bool:
    return _matches_any(chunk.text, _FACT_CHECK_PATTERNS)


def should_trigger_context_rag(chunk: TranscriptChunk, buffer: list[TranscriptChunk]) -> bool:
    """
    Context/RAG also looks at the last 3 chunks for antecedents that might
    make a pronoun-only chunk unambiguous (e.g., "Nova is fast." then "It
    supports 50k connections." — the second chunk alone doesn't trigger, but
    the window does).
    """
    if _matches_any(chunk.text, _CONTEXT_RAG_PATTERNS):
        return True
    # Sliding window: check if any of the last 3 chunks + current form a
    # pattern match on the combined text.
    recent = buffer[-3:] if len(buffer) >= 3 else buffer
    combined = " ".join(c.text for c in recent) + " " + chunk.text
    return _matches_any(combined, _CONTEXT_RAG_PATTERNS)


def should_trigger_rolling_summary(
    chunk: TranscriptChunk,
    last_summary_chunk_id: int,
    last_summary_time: float,
) -> bool:
    """Fire Rolling Summary every N chunks OR every M seconds, whichever fires first."""
    chunk_interval = settings.supervisor_summary_interval_chunks
    time_interval = settings.supervisor_summary_interval_seconds

    chunks_since_last = chunk.chunk_id - last_summary_chunk_id
    seconds_since_last = time.time() - last_summary_time

    if chunks_since_last >= chunk_interval:
        return True
    if seconds_since_last >= time_interval and last_summary_chunk_id >= 0:
        return True
    return False


def decide(
    chunk: TranscriptChunk,
    buffer: list[TranscriptChunk],
    last_summary_chunk_id: int,
    last_summary_time: float,
) -> list[AgentType]:
    """
    Return the list of AgentTypes that should be triggered for this chunk.

    Called synchronously in the ingestion WebSocket handler — must complete
    quickly. All pattern matching is O(n*m) where n=chunk length and
    m=pattern count; both are small constants.
    """
    triggered: list[AgentType] = []

    if should_trigger_action_item(chunk):
        triggered.append(AgentType.ACTION_ITEM)
        logger.debug("Supervisor: ACTION_ITEM triggered by chunk %d", chunk.chunk_id)

    if should_trigger_fact_check(chunk):
        triggered.append(AgentType.FACT_CHECK)
        logger.debug("Supervisor: FACT_CHECK triggered by chunk %d", chunk.chunk_id)

    if should_trigger_context_rag(chunk, buffer):
        triggered.append(AgentType.CONTEXT_RAG)
        logger.debug("Supervisor: CONTEXT_RAG triggered by chunk %d", chunk.chunk_id)

    if should_trigger_rolling_summary(chunk, last_summary_chunk_id, last_summary_time):
        triggered.append(AgentType.ROLLING_SUMMARY)
        logger.debug("Supervisor: ROLLING_SUMMARY triggered by chunk %d", chunk.chunk_id)

    if not triggered:
        logger.debug("Supervisor: no agents triggered for chunk %d", chunk.chunk_id)

    return triggered


def build_tasks(
    chunk: TranscriptChunk,
    buffer: list[TranscriptChunk],
    last_summary_chunk_id: int,
    last_summary_time: float,
) -> list[AgentTask]:
    """
    Convenience wrapper: run decide() and build AgentTask objects.
    The transcript buffer is shallow-copied here so agents see a stable snapshot.
    """
    triggered = decide(chunk, buffer, last_summary_chunk_id, last_summary_time)
    snapshot = list(buffer)  # O(n) copy — acceptable; buffer grows slowly
    return [
        AgentTask(
            agent_type=agent_type,
            trigger_chunk=chunk,
            transcript_snapshot=snapshot,
            session_id=chunk.session_id,
        )
        for agent_type in triggered
    ]
