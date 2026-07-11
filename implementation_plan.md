# Live Meeting Intelligence Copilot — Implementation Plan

## Background

A backend system that ingests a live meeting transcript via WebSocket, runs four specialized AI agents concurrently against the growing transcript buffer, and fans out each agent's output to all connected clients in real-time.

This plan calls out every non-trivial architectural decision as a flagged **Decision Point** (DP). Please review each DP and approve or redirect before I write code.

---

## Decision Points (DP) — Awaiting Your Review

> [!IMPORTANT]
> **DP-1: Vector Store — ChromaDB (embedded) vs. Qdrant (server)**
>
> The spec says "Chroma-backed knowledge base." I will use **ChromaDB in embedded/persistent mode** (no separate process). This is viable because the knowledge base is read-only at runtime and the process is single-node. If you want a separate Chroma server for prod-realism, say so and I'll add a Docker Compose service.

> [!IMPORTANT]
> **DP-2: Fan-out mechanism — In-process asyncio.Queue per session vs. Redis pub/sub**
>
> I am choosing **in-process `asyncio.Queue` per session with an explicit TODO comment** — not Redis — and will document **exactly why Redis would be required** (multiple gateway processes) and why that threshold isn't crossed here. DESIGN.md will cover this fully. If you want me to wire Redis pub/sub for realism, I can, but I'll call that out as scope-expansion in DESIGN.md rather than silently adding it.

> [!IMPORTANT]
> **DP-3: Agent concurrency model — `asyncio.gather` with `run_in_executor` for LangGraph nodes vs. full multi-process**
>
> LangGraph agent invocations will be dispatched via `asyncio.gather` using `loop.run_in_executor(thread_pool)`. This gives true parallelism for I/O-bound LLM calls (which release the GIL on network I/O) without the complexity of multiprocessing. A slow Fact-Check retrieval will not block Action-Item output because they run in separate executor threads with independent futures.

> [!IMPORTANT]
> **DP-4: LLM Provider — OpenAI (gpt-4o-mini) with a `.env`-configured key**
>
> I am assuming `OPENAI_API_KEY` is available. All agent calls use `gpt-4o-mini` for cost. Model is configurable via `config.py`. If you want a local Ollama fallback, I can add it, but it changes the demo performance significantly.

> [!IMPORTANT]
> **DP-5: Late joiner behavior — full history replay**
>
> When a second client joins a session mid-meeting, they receive **a single "catch-up" message** containing the current transcript buffer and all prior agent outputs accumulated so far, then switch to the live stream. The alternative (new-updates-only) was rejected because it produces a confusing UI where the late joiner sees orphaned action items with no transcript context. DESIGN.md documents this choice.

> [!IMPORTANT]
> **DP-6: Backpressure — bounded per-session work queue with drop-oldest policy**
>
> If chunks arrive faster than agents can process them, the supervisor places pending agent triggers into a bounded `asyncio.Queue(maxsize=50)`. When full, the oldest pending task is dropped (not the newest) and a warning is logged. The rationale: losing an intermediate chunk trigger is less harmful than losing the most recent one, and blocking the ingestion WebSocket is the worst outcome. DESIGN.md covers this.

> [!IMPORTANT]
> **DP-7: Frontend — plain HTML/JS (no React build step)**
>
> Using a single `frontend/index.html` with vanilla JS and `EventSource`-style WebSocket handling. No build toolchain required — the reviewer can open it in a browser directly. This satisfies "minimal but demonstrates all 4 live-updating panels."

---

## Architecture Overview

```
Browser Client(s)
       │  WebSocket /ws/{session_id}/client
       ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Application                   │
│                                                         │
│  /ws/{session_id}/ingest  ← simulator feeds chunks here │
│  /ws/{session_id}/client  ← browser clients connect     │
│                                                         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              SessionManager                       │   │
│  │  - transcript_buffer: list[TranscriptChunk]       │   │
│  │  - agent_outputs: dict[AgentType, list[Output]]   │   │
│  │  - client_queues: set[asyncio.Queue]              │   │
│  │  - work_queue: asyncio.Queue(maxsize=50)           │   │
│  └──────────────────────────────────────────────────┘   │
│                           │                             │
│              ┌────────────▼────────────┐                │
│              │       Supervisor        │                 │
│              │  heuristic-based        │                 │
│              │  trigger decision       │                 │
│              └────────────┬────────────┘                │
│                           │                             │
│         asyncio.gather (run_in_executor)                │
│    ┌──────┬──────┬──────┬──────┐                        │
│    │  FC  │  AI  │  CR  │  RS  │  ← agent threads       │
│    └──────┴──────┴──────┴──────┘                        │
│       ↓ each agent output → broadcast to client_queues  │
└─────────────────────────────────────────────────────────┘
       │
  ChromaDB (embedded persistent)
  knowledge_base/ directory
```

---

## Supervisor Triggering Heuristic (Full Detail)

The supervisor reads each new chunk and applies **four lightweight pattern-matching rules** before deciding which agents to trigger. This is O(n) regex over the chunk text — no LLM call.

| Agent | Trigger Signal | Pattern Examples |
|---|---|---|
| **Action-Item** | Commitment/assignment language | `will`, `going to`, `I'll`, `action:`, `TODO`, `by [day/date]`, `assign`, `owners` |
| **Fact-Check** | Factual assertion patterns | Numeric claims (`X% of`, `$X`, `released in YYYY`), superlatives (`fastest`, `largest`), proper nouns + `is/was/has` |
| **Context/RAG** | Ambiguous reference patterns | Pronouns (`it`, `this`, `that`) following a proper noun, `the project`, `the platform`, `last time`, `the previous` |
| **Rolling Summary** | Every Nth chunk OR time-based | Every 5 chunks OR 30 seconds since last summary run, whichever comes first |

**False-negative risk (documented):** A commitment phrased as a question ("Could Sarah maybe look into that?") or buried in a subordinate clause ("...assuming that John handles the migration...") will not trigger the Action-Item agent. This is acceptable because the system is a *copilot*, not a compliance tool — missed items can be caught in the final review, and the cost of running Action-Item on every single chunk would be 4× the LLM calls with diminishing returns on most chunks.

---

## Proposed File Structure

```
live-meeting-copilot/
├── backend/
│   ├── main.py                  # FastAPI app, WebSocket endpoints
│   ├── config.py                # env vars, model names, constants
│   ├── session_manager.py       # SessionManager class, session registry
│   ├── supervisor.py            # heuristic trigger logic
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py              # AgentOutput dataclass, base interface
│   │   ├── fact_check.py        # Fact-Check LangGraph node
│   │   ├── action_item.py       # Action-Item LangGraph node
│   │   ├── context_rag.py       # Context/RAG LangGraph node
│   │   └── rolling_summary.py   # Rolling Summary LangGraph node
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── chroma_store.py      # ChromaDB wrapper, retrieval helpers
│   │   └── ingest_docs.py       # One-time script to load sample docs
│   └── models.py                # Pydantic models for WS messages
├── simulator/
│   └── transcript_simulator.py  # Feeds sample transcript chunk-by-chunk
├── knowledge_base/
│   ├── sample_docs/             # 6–8 plain-text knowledge docs
│   └── chroma_db/               # ChromaDB persistent storage (gitignored)
├── frontend/
│   └── index.html               # Vanilla JS, 5-panel live UI
├── transcript_samples/
│   └── meeting_transcript.txt   # Sample meeting with facts, AIs, ambiguous refs
├── DESIGN.md                    # Architecture decisions + tradeoffs
├── README.md                    # Setup + run instructions
├── requirements.txt
└── .env.example
```

---

## Proposed Changes — Component by Component

### Component 1: Data Models & Config

#### [NEW] [config.py](file:///c:/Users/KIIT0001/Desktop/live%20meeting%20project/backend/config.py)
- `OPENAI_API_KEY`, `LLM_MODEL`, `CHROMA_PATH`, `SUPERVISOR_SUMMARY_INTERVAL`, `WORK_QUEUE_MAXSIZE`

#### [NEW] [models.py](file:///c:/Users/KIIT0001/Desktop/live%20meeting%20project/backend/models.py)
- `TranscriptChunk`: `session_id`, `chunk_id`, `speaker`, `text`, `timestamp`
- `AgentOutput`: `session_id`, `agent_type`, `chunk_id`, `content`, `citations: list[str]`, `timestamp`
- `WSMessage`: union type for `chunk_ack`, `agent_output`, `catchup`, `error`

---

### Component 2: Session Manager

#### [NEW] [session_manager.py](file:///c:/Users/KIIT0001/Desktop/live%20meeting%20project/backend/session_manager.py)
- `SessionState`: holds `transcript_buffer`, `agent_output_log`, `client_queues: set[asyncio.Queue]`, `work_queue: asyncio.Queue(maxsize=50)`, `last_chunk_id: int`
- `SessionRegistry`: singleton dict `{session_id: SessionState}`, creates sessions on first connect
- `broadcast(session_id, msg)`: puts msg into every client queue non-blockingly; drops silently only if a client queue is itself full (client-side backpressure, not session-wide)
- `catchup_payload(session_id)`: returns full transcript + all prior agent outputs for late joiners

---

### Component 3: FastAPI WebSocket Endpoints

#### [NEW] [main.py](file:///c:/Users/KIIT0001/Desktop/live%20meeting%20project/backend/main.py)
- `POST /sessions` — create session, return `session_id`
- `WS /ws/{session_id}/ingest` — accepts `TranscriptChunk` JSON, validates chunk order (rejects if `chunk_id <= last_chunk_id`), appends to buffer, hands to supervisor
- `WS /ws/{session_id}/client` — on connect: sends catchup payload, adds queue to session; drains queue to client; on disconnect: removes queue cleanly
- `GET /sessions/{session_id}/state` — REST endpoint for debugging (returns current transcript + outputs)

**Malformed chunk handling:** `try/except ValidationError` around `TranscriptChunk.model_validate(data)`. Sends `{type: "error", detail: "..."}` back to ingestor, does NOT append to buffer, does NOT crash session.

**Out-of-order chunk handling:** If `chunk.chunk_id != session.last_chunk_id + 1`, logs a warning and sends an `out_of_order` error. Does not corrupt buffer. (Assumption: simulator is reliable; this is a guard, not a full reorder buffer. Documented in DESIGN.md.)

---

### Component 4: Supervisor

#### [NEW] [supervisor.py](file:///c:/Users/KIIT0001/Desktop/live%20meeting%20project/backend/supervisor.py)
- `should_trigger(agent_type, chunk, buffer) -> bool` — pure function, no side effects
- Pattern sets compiled at import time (no per-call compile)
- `decide(chunk, buffer, last_summary_chunk) -> list[AgentType]`
- Places `AgentTask(agent_type, chunk_snapshot, buffer_snapshot)` into `work_queue`
- Worker coroutine `_agent_worker` drains the work queue, dispatches agents via `asyncio.gather` with `run_in_executor`

---

### Component 5: Agents

Each agent is a standalone async function `run(task: AgentTask) -> AgentOutput`. LangGraph is used as the **graph/node composition layer** for the Fact-Check and Context agents (which have a retrieve → reason → cite multi-step flow). Action-Item and Rolling Summary use a single LangGraph node (trivial graph) since they don't need retrieval branching.

#### [NEW] [agents/fact_check.py](file:///c:/Users/KIIT0001/Desktop/live%20meeting%20project/backend/agents/fact_check.py)
- Graph: `extract_claims` → `retrieve_docs` → `cross_reference` → `emit`
- `retrieve_docs`: calls `chroma_store.similarity_search(claim, k=3)`
- `cross_reference`: LLM prompt includes retrieved doc excerpts and source names; output **must** include `citations`
- If no docs retrieved: output `{"verdict": "unverifiable", "reason": "No relevant documents in knowledge base"}` — never silent

#### [NEW] [agents/action_item.py](file:///c:/Users/KIIT0001/Desktop/live%20meeting%20project/backend/agents/action_item.py)
- Single LangGraph node
- Prompt: extract `[{assignee, task, deadline_if_mentioned}]` from chunk + last 5 transcript lines for speaker context
- Returns structured JSON list

#### [NEW] [agents/context_rag.py](file:///c:/Users/KIIT0001/Desktop/live%20meeting%20project/backend/agents/context_rag.py)
- Graph: `extract_references` → `retrieve_docs` → `resolve` → `emit`
- Similar to Fact-Check but resolving pronouns/ambiguous terms rather than verifying facts
- Citations: same format, document name + excerpt

#### [NEW] [agents/rolling_summary.py](file:///c:/Users/KIIT0001/Desktop/live%20meeting%20project/backend/agents/rolling_summary.py)
- Maintains a **running summary state** in `SessionState`
- Each run: LLM takes `(previous_summary, new_chunks_since_last_summary)` → produces updated summary
- NOT a one-shot full-transcript summarize — always incremental

---

### Component 6: RAG / ChromaDB

#### [NEW] [rag/chroma_store.py](file:///c:/Users/KIIT0001/Desktop/live%20meeting%20project/backend/rag/chroma_store.py)
- Singleton `ChromaStore` initialized at startup
- `similarity_search(query, k) -> list[Document]` where `Document` has `.page_content` and `.metadata["source"]`
- Returns empty list (not raises) when no results above threshold

#### [NEW] [rag/ingest_docs.py](file:///c:/Users/KIIT0001/Desktop/live%20meeting%20project/backend/rag/ingest_docs.py)
- One-time script: reads all `.txt` files from `knowledge_base/sample_docs/`, chunks them, embeds, stores in ChromaDB

#### [NEW] knowledge_base/sample_docs/ (6–8 docs)
- `company_products.txt` — product names, release dates, specs (for Fact-Check to verify)
- `engineering_team.txt` — team members, roles (for Context/RAG to resolve "the platform team")
- `q2_roadmap.txt` — current roadmap items (for Fact-Check on milestone claims)
- `competitors.txt` — competitor facts (for Fact-Check)
- `meeting_history.txt` — prior meeting summaries (for Context to resolve "last time we discussed")
- `technical_glossary.txt` — internal project/term definitions

---

### Component 7: Transcript Simulator

#### [NEW] [simulator/transcript_simulator.py](file:///c:/Users/KIIT0001/Desktop/live%20meeting%20project/simulator/transcript_simulator.py)
- Reads `transcript_samples/meeting_transcript.txt`
- Parses into `TranscriptChunk` JSON per line (format: `SPEAKER: text`)
- Sends via `websockets` library to `ws://localhost:8000/ws/{session_id}/ingest`
- Configurable `--delay` (default 1.5s between chunks) and `--session-id`
- On completion: sends a final `{type: "meeting_end"}` message

#### [NEW] transcript_samples/meeting_transcript.txt
- ~40–50 lines, scripted to contain:
  - Verifiable factual claims (matching knowledge base docs)
  - False/contradicting claims (Fact-Check should flag)
  - Unverifiable claims (no KB doc covers them)
  - Action items with assignees and deadlines
  - Ambiguous pronoun references
  - Natural meeting language throughout

---

### Component 8: Frontend

#### [NEW] frontend/index.html
- Two WebSocket connections in a single page (two `<iframe>`s or one connection with two tabs) to demonstrate fan-out
- Five panels: Live Transcript | Action Items | Fact-Check Flags | Context Cards | Rolling Summary
- Each panel independently receives and renders its type of agent output
- Color-coded: green = verified, red = contradiction, yellow = unverifiable
- No build step; pure HTML + JS + CSS in one file

---

### Component 9: Documentation

#### [NEW] DESIGN.md
Sections:
1. Supervisor Heuristic + False-Negative Tradeoff
2. Fan-out: In-Process vs. Redis (explicit "this breaks at N>1 process" statement)
3. Backpressure: Bounded queue + drop-oldest policy + why not block-ingestor
4. Late Joiner: Full history replay rationale
5. Out-of-Order Chunks: Guard vs. full reorder buffer decision
6. "Unverifiable" vs. silent: Fact-Check policy
7. Concurrency model: asyncio + executor vs. multiprocessing

#### [NEW] README.md
- Prerequisites (Python 3.11+, OpenAI key)
- `pip install -r requirements.txt`
- `python backend/rag/ingest_docs.py` (one-time)
- `uvicorn backend.main:app --reload`
- `python simulator/transcript_simulator.py --session-id demo-1`
- Open `frontend/index.html` in two browser windows/tabs

---

## Verification Plan

### Automated / Demonstrable
1. Run the ingestor + two separate browser tabs on the same `session_id` → both must show identical updates in real-time (browser recording)
2. Send a malformed JSON chunk via `wscat` → server returns error, does not close session, next valid chunk is accepted
3. Open second browser tab mid-session → confirm catchup payload appears before live updates
4. Inspect server logs: verify Fact-Check and Action-Item sometimes run concurrently (log timestamps overlap) and sometimes one runs without the other

### Manual Review
- DESIGN.md reviewed against each spec requirement
- Fact-Check output for a claim that contradicts KB shows the contradicting document name
- Fact-Check output for an unknown claim shows `"unverifiable"`, not silence
- Rolling summary updates visibly after every 5 chunks, not only at meeting end

---

## Open Questions

> [!WARNING]
> **Q1**: Should I implement Redis pub/sub for real (requiring Docker Compose) or implement in-process and document Redis as the production path? My recommendation: in-process + DESIGN.md rationale. Please confirm.

> [!WARNING]
> **Q2**: The spec says "LangGraph nodes/subgraphs." Should I use LangGraph's actual `StateGraph` with typed state for every agent, or use LangGraph for the two multi-step agents (Fact-Check, Context) and a simpler async function for the single-step ones (Action-Item, Rolling Summary)? My recommendation: `StateGraph` for all four for consistency, even if trivial for the single-step ones.

> [!WARNING]
> **Q3**: Embedding model — `text-embedding-3-small` (OpenAI, requires API calls) vs. `sentence-transformers/all-MiniLM-L6-v2` (local, free, slower first load). My recommendation: `text-embedding-3-small` for demo quality and consistency with the LLM provider.

> [!WARNING]
> **Q4**: The spec mentions a "recorded browser verification." Should I use the `browser_subagent` tool to record this after the system is built, or is a written confirmation of the behavior sufficient for the plan approval stage?

