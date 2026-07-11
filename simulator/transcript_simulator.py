"""
Transcript Simulator — feeds a sample transcript file to the server chunk-by-chunk.

Usage:
    python simulator/transcript_simulator.py --session-id demo-1 --delay 2.0
    python simulator/transcript_simulator.py --session-id demo-1 --delay 1.5 --url ws://localhost:8000

Arguments:
    --session-id   Session ID to stream into (default: demo-1)
    --delay        Seconds between chunks (default: 2.0)
    --url          Server base WebSocket URL (default: ws://localhost:8000)
    --transcript   Path to transcript file (default: transcript_samples/meeting_transcript.txt)

The simulator:
    1. Connects to /ws/{session_id}/ingest
    2. Reads the transcript file line by line
    3. Parses "SPEAKER: text" format into TranscriptChunk JSON
    4. Sends each chunk with a configurable delay between lines
    5. Sends a meeting_end signal when the file is exhausted
    6. Prints server acknowledgements to stdout for visibility
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# Allow running as: python simulator/transcript_simulator.py
# from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import websockets
    from websockets.exceptions import ConnectionClosed
except ImportError:
    print("ERROR: websockets package not installed. Run: pip install -r requirements.txt")
    sys.exit(1)


def parse_transcript(file_path: Path) -> list[dict]:
    """
    Parse the transcript file into a list of chunk dicts.

    Each non-empty line must be in the format:  SPEAKER: text
    Lines that don't match this format are skipped with a warning.
    """
    chunks = []
    chunk_id = 0

    with open(file_path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                print(f"  WARN: Line {line_no} skipped (no colon separator): {line!r}")
                continue
            speaker, _, text = line.partition(":")
            speaker = speaker.strip()
            text = text.strip()
            if not speaker or not text:
                print(f"  WARN: Line {line_no} skipped (empty speaker or text)")
                continue
            chunks.append({
                "chunk_id": chunk_id,
                "speaker": speaker,
                "text": text,
                "timestamp": time.time(),
            })
            chunk_id += 1

    return chunks


async def run_simulator(
    session_id: str,
    delay: float,
    server_url: str,
    transcript_path: Path,
) -> None:
    ingest_url = f"{server_url}/ws/{session_id}/ingest"

    print(f"\n{'='*60}")
    print(f"Live Meeting Intelligence Copilot — Transcript Simulator")
    print(f"{'='*60}")
    print(f"Session ID   : {session_id}")
    print(f"Server URL   : {ingest_url}")
    print(f"Transcript   : {transcript_path}")
    print(f"Chunk delay  : {delay}s")
    print(f"{'='*60}\n")

    # Parse transcript before connecting.
    if not transcript_path.exists():
        print(f"ERROR: Transcript file not found: {transcript_path}")
        sys.exit(1)

    chunks = parse_transcript(transcript_path)
    if not chunks:
        print("ERROR: No valid chunks found in transcript file.")
        sys.exit(1)

    print(f"Parsed {len(chunks)} chunks. Connecting to server…\n")

    try:
        async with websockets.connect(ingest_url) as ws:
            print("✓ Connected to ingest endpoint.\n")

            for i, chunk_data in enumerate(chunks):
                # Send chunk.
                payload = json.dumps(chunk_data)
                await ws.send(payload)

                # Print what we sent.
                print(
                    f"[{i+1:02d}/{len(chunks)}] "
                    f"{chunk_data['speaker']:12s} → {chunk_data['text'][:60]}{'…' if len(chunk_data['text']) > 60 else ''}"
                )

                # Wait for server ack.
                try:
                    ack_raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
                    ack = json.loads(ack_raw)
                    if ack.get("type") == "chunk_ack":
                        triggered = ack.get("agents_triggered", [])
                        if triggered:
                            print(f"             ↳ Agents triggered: {', '.join(triggered)}")
                    elif ack.get("type") == "error":
                        print(f"             ↳ SERVER ERROR: {ack.get('detail', ack)}")
                    elif ack.get("type") == "ping":
                        pass  # Ignore ping from client endpoint bleed-through
                except asyncio.TimeoutError:
                    print("             ↳ WARNING: No ack received within 10s")

                # Wait before sending next chunk (simulating real speech pace).
                if i < len(chunks) - 1:
                    await asyncio.sleep(delay)

            # Signal meeting end.
            end_signal = json.dumps({"type": "meeting_end", "session_id": session_id})
            await ws.send(end_signal)
            print(f"\n[END] Meeting end signal sent.")

            # Wait for end ack.
            try:
                end_ack = await asyncio.wait_for(ws.recv(), timeout=5.0)
                print(f"[END] Server ack: {end_ack}")
            except asyncio.TimeoutError:
                print("[END] No ack for meeting_end (timeout).")

    except ConnectionClosed as exc:
        print(f"\nERROR: Connection closed unexpectedly: {exc}")
        sys.exit(1)
    except OSError as exc:
        print(f"\nERROR: Could not connect to {ingest_url}: {exc}")
        print("Is the server running? Try: uvicorn backend.main:app --reload")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Simulation complete. {len(chunks)} chunks sent.")
    print(f"{'='*60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feed a transcript file to the Live Meeting Copilot server."
    )
    parser.add_argument(
        "--session-id",
        default="demo-1",
        help="Session ID to stream into (default: demo-1)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds between transcript chunks (default: 2.0)",
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:8000",
        help="Server base WebSocket URL (default: ws://localhost:8000)",
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=PROJECT_ROOT / "transcript_samples" / "meeting_transcript.txt",
        help="Path to transcript .txt file",
    )
    args = parser.parse_args()

    asyncio.run(
        run_simulator(
            session_id=args.session_id,
            delay=args.delay,
            server_url=args.url,
            transcript_path=args.transcript,
        )
    )


if __name__ == "__main__":
    main()
