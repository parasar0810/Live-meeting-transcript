# Live Meeting Intelligence Copilot — Design Document

This document captures every non-obvious architectural decision in the system. It is written to defend these choices in a technical design review, not to describe what was built.

---

## 1. Supervisor Triggering Heuristic

### Decision

The supervisor uses compiled regex patterns evaluated once per incoming chunk. No LLM call is made in the triggering path. This was a deliberate rejection of the "ask the LLM to decide" approach.

### Rationale

LLM-based routing has two problems at this scale:
1. **Latency**: A GPT-4o-mini call takes 300–800ms. If every chunk triggers a routing call before agents can run, a 2-second chunk interval becomes a 2.3–2.8s round-trip before useful work starts. Under burst conditions (chunks arriving faster than the routing LLM responds), backpressure would build against the routing layer itself.
2. **Cost multiplication**: With 4 agents, routing via LLM adds a 5th LLM call (the router) to every chunk, regardless of whether any agent ends up running. For a 40-chunk meeting, that's 40 router calls vs. 0 with regex.

The regex patterns were designed from analysis of meeting transcript patterns. They are compiled at import time (O(1) compile cost) and evaluated per chunk at O(n×m) where n is chunk character count and m is pattern count — both small constants in practice.

### Per-Agent Trigger Logic

| Agent | Trigger signal | Compiled pattern examples |
|---|---|---|
| **Action-Item** | Commitment verbs, assignment language, deadline markers | `I'll`, `we will`, `by Friday`, `action item:`, `needs to` |
| **Fact-Check** | Numeric claims, superlatives, product-name assertions, compliance terms | `\d+%`, `$\d+`, `fastest/largest`, `SOC2`, `MTTD`, `P99` |
| **Context/RAG** | Ambiguous pronouns after proper nouns, "the X" references, prior-discussion references | `the (platform\|project\|migration)`, `it/this/that` near product names, `as discussed` |
| **Rolling Summary** | Time/chunk count threshold | Every 5 chunks OR 30 seconds since last run |

### False-Negative Risks and Acceptability

**Action-Item agent false negatives:**
- *Missed trigger*: Commitments phrased as questions: "Could Sarah maybe look into the database issue?" — the words `I'll`, `will`, `going to`, `needs to` are absent.
- *Missed trigger*: Passive-voice commitments: "It was decided that Jordan would handle the migration."
- *Missed trigger*: Implicit subject: "Let's get that done by Friday" (no named assignee in the chunk).
- *Acceptability*: This system is a copilot, not a compliance recorder. An action item missed by the live copilot will be caught in the final review of the meeting notes. The cost of running Action-Item on every chunk to catch these edge cases is 4× the LLM calls with diminishing returns. The heuristic is tuned for precision over recall by design.

**Fact-Check agent false negatives:**
- *Missed trigger*: Claims as rhetorical questions: "Aren't we already at 100k connections?"
- *Missed trigger*: Claims in subordinate clauses without numeric/superlative markers: "...assuming that StreamCore already certified for ISO 27001..."
- *Acceptability*: The same argument applies — the copilot surfaces high-signal claims for real-time verification. Low-confidence phrasing (rhetorical questions, subordinate clauses) is less likely to be asserted as fact in a meeting context.

**Context/RAG agent false negatives:**
- *Missed trigger*: Cross-chunk anaphora where the pronoun and antecedent span more than 3 chunks.
- *Missed trigger*: Domain-specific abbreviations not in the trigger list.
- *Acceptability*: The sliding window (last 3 chunks + current) mitigates most cross-chunk pronoun cases. Unknown abbreviations are a known gap; the pattern list is the correct place to extend this over time.

**Rolling Summary: no false negatives.** It fires on a schedule regardless of content.

---

## 2. Fan-Out Mechanism: In-Process vs. Redis

### Decision

In-process `asyncio.Queue` per connected client, coordinated by a module-level `SessionRegistry`.

### Why Not Redis Pub/Sub

Redis pub/sub is required when clients may be connected to **different server processes** that cannot share memory. This system documents its setup as a single Uvicorn process (`uvicorn backend.main:app --workers 1`). In that configuration:
- All WebSocket connections share the same Python process and event loop.
- `asyncio.Queue` objects are in-process; they have zero serialization overhead and sub-microsecond write latency.
- Redis pub/sub would require JSON serialization per message, network round-trip to Redis, and deserialization per subscriber — all for messages that already exist in process memory.

**The exact failure mode at N>1 processes:**
If you run `uvicorn backend.main:app --workers 4` (or behind Gunicorn with 4 workers), Client A on Worker 1 and Client B on Worker 2 cannot be reached by a broadcast call on Worker 3. `broadcast()` only knows about queues in its own process's `SessionRegistry`. The second client would receive no live updates.

**Migration path**: replace `broadcast()` in `session_manager.py` with a Redis pub/sub publish call and add a per-session Redis subscriber per client WebSocket. The `SessionRegistry` becomes a thin wrapper with no in-process queue sharing. The rest of the system is unchanged.

### Why Each Client Has Its Own Queue

An alternative is a single session-level queue with a fan-out consumer. The problem: if the consumer blocks sending to one slow client, all clients stall. Per-client queues let a slow client's queue fill up and drop silently without affecting other clients (see `broadcast()` — it uses `put_nowait()` and catches `QueueFull`).

---

## 3. Backpressure Strategy

### Decision

Bounded `asyncio.Queue(maxsize=50)` per session for agent work tasks. When full, the **oldest item is evicted** before the new one is inserted.

### Why Drop-Oldest, Not Drop-Newest

The newest chunk is the most recent context. Losing a trigger for an older chunk means missing an action item or fact-check that was already somewhat distant from the current discussion. Losing a trigger for the newest chunk means the system is behind on the most actionable content.

### Why Not Block the Ingestor

Blocking the ingestion WebSocket is the worst possible outcome: it causes the simulator (and, in production, the real audio transcription feed) to stall, backing pressure up to the audio capture layer. The transcript buffer would freeze. This is worse than losing a few intermediate agent triggers.

### Why 50 Items

At 4 agents per chunk max × 2-second chunk interval, maximum agent trigger rate is ~2 per second. A 50-item queue gives ~25 seconds of runway before eviction begins. In practice, agents rarely all trigger simultaneously and most complete within 5–10 seconds; the queue stays near-empty under normal conditions.

### Client-Side Backpressure

Per-client queues are also bounded (`CLIENT_QUEUE_MAXSIZE=100`). A slow browser client (e.g., tab in background) cannot stall the broadcast path — its queue fills and overflow messages are dropped silently with a warning log.

---

## 4. Late Joiner Policy: Full History Replay

### Decision

When a client connects to an ongoing session, it receives a `catchup` message containing the full transcript buffer and all prior agent outputs produced since session start.

### Why Full History

The alternative — send only new updates — produces an incoherent UX: the late joiner sees action items without the transcript lines that prompted them, and fact-check flags referencing claims they've never seen. The copilot becomes useless without context.

### Cost

The catch-up payload can be large for a long meeting (40 transcript lines + 20+ agent outputs). This is a single WebSocket send of JSON — typically 5–50 KB, which is negligible at modern network speeds. There is no reason to page this or send it incrementally.

### Implementation

`build_catchup()` in `session_manager.py` takes a snapshot of `session.transcript` and `session.agent_output_log` at connect time. The snapshot is taken atomically (Python list copy under the GIL). Messages broadcast while `build_catchup` executes are queued in the client's queue and will be delivered immediately after the catch-up send completes, maintaining order.

---

## 5. Out-of-Order Chunk Handling

### Decision

The server is a **guard**, not a reorder buffer. If a chunk arrives out of order (gap in chunk_id sequence), it is accepted with a warning and the `last_chunk_id` is advanced. A full reorder buffer was explicitly rejected.

### Rationale

A reorder buffer (holding chunks until the missing predecessor arrives) adds complexity and a correctness dependency: if the missing chunk never arrives (simulator crash, network drop), the buffer deadlocks and the session freezes. For a demo-quality system with a reliable local simulator, this tradeoff is unacceptable. The guard approach ensures the session always makes progress, accepting the theoretical risk of rare ordering inconsistency in the transcript buffer.

**When a reorder buffer would be required:** a production deployment with untrusted real-time audio transcription over unreliable network, where chunks genuinely arrive out of order at non-trivial rates.

Duplicate chunks (chunk_id ≤ last_chunk_id) are silently discarded and an error is returned to the ingestor.

---

## 6. Unverifiable vs. Silence in Fact-Check

### Decision

When the Fact-Check agent finds no relevant documents in the knowledge base for a claim, it emits a result with `verdict=UNVERIFIABLE` and `explanation="No relevant documents found in the knowledge base to verify this claim."` It never returns an empty payload for a verified claim trigger.

### Rationale

Silence is indistinguishable from "the agent didn't run." If Fact-Check runs and finds nothing, the user needs to know:
1. That the agent ran and actively checked.
2. That the knowledge base doesn't cover this claim — this is itself useful signal (the claim may be about external data not in the KB, or may be an invented fact).

An UNVERIFIABLE verdict is explicitly more useful than silence because it prompts the user to manually verify the claim, rather than implicitly trusting it because the copilot said nothing.

The only case where Fact-Check emits no result is when the chunk contains **no verifiable claims** (the extract_claims LLM node returns []). In that case, the agent still ran — the supervisor was triggered by a false-positive pattern match — but there is genuinely nothing to check.

---

## 7. Agent Concurrency Model

### Decision

Agent functions are `async` and dispatched via `asyncio.gather`. Blocking LLM and ChromaDB calls within agents are wrapped in `asyncio.to_thread` (equivalent to `loop.run_in_executor`). A module-level `ThreadPoolExecutor` with 8 workers handles the blocking calls.

### Why asyncio + executor, not multiprocessing

**Multiprocessing** would require serializing all agent state (transcript snapshot, LangGraph state) across process boundaries, and would not share the ChromaDB embedded client cleanly (it holds file locks). The benefit — true CPU-level parallelism — does not apply here because all the slow operations are network I/O (OpenAI API calls), not CPU-bound. Network I/O releases the Python GIL, so threads achieve the same effective parallelism.

**asyncio.gather** is the correct primitive for starting multiple agents at once. Each `gather` call starts all coroutines before awaiting any of them. If Fact-Check takes 4 seconds and Action-Item takes 1 second, Action-Item's result is broadcast at t+1s, not t+5s. This is the core correctness property for the "agents don't block each other" requirement.

**Demonstration** (visible in server logs at INFO level):
```
INFO agent-worker-demo-1 — Running 2 agent(s) concurrently for chunk 5: ['fact_check', 'action_item']
INFO agent-worker-demo-1 — Agent action_item completed for chunk 5 in 1.23s
INFO agent-worker-demo-1 — Agent fact_check completed for chunk 5 in 3.87s
```
Action-Item's output reaches clients at t+1.23s; Fact-Check's at t+3.87s. They are independent.

### LangGraph Usage

All four agents use LangGraph `StateGraph`. The two-step agents (Action-Item, Rolling Summary) use trivially short graphs (`node → emit → END`) for architectural consistency — the reviewer can inspect all agents the same way, and adding nodes to any agent is a one-line graph change.

---

## 8. Vector Store: ChromaDB Embedded Mode

### Decision

ChromaDB persistent embedded mode (no separate server process). Collection is pre-populated by `ingest_docs.py` before the server starts and read-only at runtime.

### Rationale

A separate Chroma server (Docker Compose) adds operational complexity with no benefit for a single-node read-only workload. The embedded client holds file locks on the persistence directory; since only one process reads at runtime and all reads are from the main Uvicorn process, there are no lock conflicts.

**When a Chroma server would be required:** multiple Uvicorn workers each needing to read the same collection. The embedded client does not support multi-process concurrent access safely.

### Embedding Model

194: `text-embedding-3-small` (OpenAI). Chosen over local sentence-transformers for:
195: 1. Quality: dimensionality and model capacity is higher than MiniLM.
196: 2. Consistency: same API provider as the LLM; no second dependency.
197: 3. Speed: OpenAI's embedding endpoint is fast enough that retrieval latency is dominated by the similarity search, not the embedding call.
198: 
199: Cost: ~$0.002 per 1M tokens. The knowledge base is ~6 documents × ~1000 tokens = ~6000 tokens to embed. Negligible.
200: 
201: ---
202: 
203: ## 9. LLM Failure Handling and Rate Limits
204: 
205: ### Observed Behavior (Post-QA Pass)
206: 
207: When an LLM API call fails (e.g., due to rate limits from bursting chunks or a bad API key), the agent execution function throws an exception. `run_tasks_concurrent` in `backend/agents/base.py` catches this exception, logs the error, and returns `None`. 
208: 
209: ### Decision
210: 
211: The system **silently drops** the failed agent's output instead of sending an error message to the client or implementing exponential backoff retries.
212: 
213: ### Rationale
214: 
215: 1. **Client noise**: Sending error payloads for transient rate limits creates a messy user experience (e.g., red flashing "Error" cards in the UI).
216: 2. **Staleness**: In a live meeting context, if an Action-Item extraction is delayed by 30 seconds of exponential backoff retries, it loses its value as a real-time copilot. Dropping the output and waiting for the next chunk is a better tradeoff than delivering stale insights out of order.
217: 3. **System stability**: Crashing the background worker task on LLM failure would kill the session for all connected clients. Returning `None` isolates the fault to a single dropped insight, allowing the session to process the next chunk immediately.
218: 
219: **Known Limitation**: In environments with strict LLM rate limits (like free-tier keys), bursts of incoming chunks will result in zero visible agent outputs, degrading the copilot's utility entirely. Production deployments must provision sufficient LLM throughput to avoid this.
