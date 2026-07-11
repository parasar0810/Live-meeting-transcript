# Live Meeting Intelligence Copilot

<video src="https://github.com/parasar0810/Live-meeting-transcript/raw/main/demo.mp4" controls="controls" style="max-width: 100%;"></video>

A backend that ingests a live meeting transcript over WebSocket, runs four concurrent specialized AI agents against the growing transcript, and fans out each agent's output to all connected browser clients in real time.

## Architecture at a Glance

```
Browser Client(s)  ←── WebSocket /ws/{session_id}/client
                             ↑
            FastAPI + per-session asyncio worker
                             ↑
Transcript Simulator  ──→  WebSocket /ws/{session_id}/ingest
                             ↓
                    Supervisor (regex heuristics)
                    decides which agents to trigger
                             ↓
          asyncio.gather ─── 4 concurrent LangGraph agents
          ├── Fact-Check     (ChromaDB retrieval + cite source)
          ├── Action-Item    (extract assignee + deadline)
          ├── Context/RAG    (resolve ambiguous references)
          └── Rolling Summary (incremental, not final-only)
                             ↓
                    Broadcast to all clients
```

## Prerequisites

- Python 3.11+
- An OpenAI API key

## Setup

### 1. Install dependencies

```bash
cd "live meeting project"
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

### 3. Ingest the knowledge base (one-time)

This populates ChromaDB with the 6 sample knowledge-base documents. Run from the project root:

```bash
python -m backend.rag.ingest_docs
```

Expected output:
```
INFO: Found 6 documents to ingest
INFO: Ingestion complete. 87 chunks added/updated.
```

## Running the System

### Step 1 — Start the server

```bash
uvicorn backend.main:app --reload --port 8000
```

You should see:
```
INFO: Starting up — initialising ChromaDB…
INFO: ChromaDB ready — collection 'meeting_kb' has 87 document chunks
INFO: Startup complete.
```

### Step 2 — Open the frontend (two browser windows)

Open `frontend/index.html` in **two separate browser windows or tabs**. In each:
1. Enter session ID: `demo-1`
2. Click **Connect**

Both clients should show "Connected · demo-1" in the status bar.

> **Tip**: Open the second window after the simulator has sent a few chunks to demonstrate the late-joiner catch-up feature.

### Step 3 — Run the transcript simulator

```bash
python simulator/transcript_simulator.py --session-id demo-1 --delay 2.0
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--session-id` | `demo-1` | Session ID to stream into |
| `--delay` | `2.0` | Seconds between chunks (lower = faster demo) |
| `--url` | `ws://localhost:8000` | Server WebSocket base URL |
| `--transcript` | `transcript_samples/meeting_transcript.txt` | Transcript file path |

**Fast demo mode** (for impatient demos):
```bash
python simulator/transcript_simulator.py --session-id demo-1 --delay 0.8
```

## What to Observe

### During the simulation

| Panel | What you'll see |
|---|---|
| **Live Transcript** | Each line appears as it's sent, labeled by speaker |
| **Action Items** | Cards appear when commitment language is detected (e.g., "Jordan will send the API spec by May 17th") |
| **Fact-Check** | Green ✅ for verified claims, Red ❌ for contradictions (e.g., "StreamCore supports 100k connections" → CONTRADICTION), Yellow ⚠️ for unverifiable |
| **Context Resolution** | "The migration" → "Python 2→3 threat detection pipeline", citing `meeting_history` |
| **Rolling Summary** | Updates every 5 chunks — watch it evolve, not just appear at the end |

### Agent concurrency in server logs

Watch the server terminal — you'll see log lines like:
```
INFO  Running 2 agent(s) concurrently for chunk 5: ['fact_check', 'action_item']
INFO  Agent action_item completed for chunk 5 in 1.23s
INFO  Agent fact_check completed for chunk 5 in 3.87s
```
Action-Item's output reaches both browser clients at t+1.23s independently of Fact-Check.

### Late joiner test

1. Start the simulator with `--delay 3.0`
2. After 5–10 chunks, open a **second browser window** and connect to `demo-1`
3. The second client immediately receives a catch-up payload (prior transcript + all agent outputs), then switches to the live stream seamlessly

### Malformed chunk test

```bash
# In a separate terminal, using wscat or similar:
wscat -c ws://localhost:8000/ws/demo-1/ingest
> {"chunk_id": "notanumber", "speaker": "SARAH", "text": "This is malformed"}
```
Server returns `{"type":"error","code":"validation_error",...}` and the session continues normally.

## Project Structure

```
live-meeting-copilot/
├── backend/
│   ├── main.py              # FastAPI app, WebSocket endpoints
│   ├── config.py            # All settings via .env
│   ├── models.py            # Pydantic message schemas
│   ├── session_manager.py   # Session state, fan-out, backpressure
│   ├── supervisor.py        # Heuristic trigger decision (no LLM)
│   └── agents/
│       ├── base.py          # Concurrent runner, agent registry
│       ├── fact_check.py    # LangGraph: extract→retrieve→verify→cite
│       ├── action_item.py   # LangGraph: extract commitments+speaker
│       ├── context_rag.py   # LangGraph: extract refs→retrieve→resolve
│       └── rolling_summary.py # LangGraph: incremental summary
├── backend/rag/
│   ├── chroma_store.py      # ChromaDB singleton wrapper
│   └── ingest_docs.py       # One-time KB ingestion script
├── simulator/
│   └── transcript_simulator.py
├── knowledge_base/
│   └── sample_docs/         # 6 plain-text KB documents
├── transcript_samples/
│   └── meeting_transcript.txt  # 42-line scripted meeting
├── frontend/
│   └── index.html           # Vanilla JS, 5-panel live UI
├── DESIGN.md                # All architectural decisions + tradeoffs
├── requirements.txt
└── .env.example
```

## Key Design Decisions (summary — see DESIGN.md for full rationale)

| Decision | Choice | Why |
|---|---|---|
| Supervisor | Regex patterns | No per-chunk LLM call; O(1) compile, fast evaluation |
| Fan-out | In-process asyncio.Queue | Single process; Redis needed only for N>1 workers |
| Backpressure | Bounded queue, drop-oldest | Never block ingestor; newest context is most valuable |
| Late joiner | Full history replay | Partial history = incoherent UX |
| Agent concurrency | asyncio.gather + to_thread | I/O-bound LLM calls; threads release GIL |
| "No KB doc" verdict | UNVERIFIABLE, not silence | Silence ≡ agent didn't run; UNVERIFIABLE is actionable |
| Vector store | ChromaDB embedded | Single-node read-only; no separate process needed |
| Embeddings | text-embedding-3-small | Higher quality than local models; same API provider |

## Troubleshooting

**`OPENAI_API_KEY` not found**: Ensure your `.env` file exists in the project root (same directory as `requirements.txt`) and contains `OPENAI_API_KEY=sk-...`.

**"Knowledge base is empty"**: Run `python -m backend.rag.ingest_docs` from the project root before starting the server.

**Simulator can't connect**: Ensure the server is running at `http://localhost:8000`. Check with `curl http://localhost:8000/health`.

**Frontend shows no updates**: Open browser DevTools → Console. If you see WebSocket errors, check that the session ID in the frontend matches the `--session-id` flag used by the simulator.
