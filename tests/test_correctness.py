import asyncio
import json
import uuid
import websockets

async def test_correctness():
    session_id = f"test-correctness-{uuid.uuid4().hex[:8]}"
    ingest_uri = f"ws://localhost:8000/ws/{session_id}/ingest"
    client_uri = f"ws://localhost:8000/ws/{session_id}/client"
    
    print(f"Starting correctness test for session {session_id}")
    
    with open("transcript_samples/meeting_transcript.txt", "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    async with websockets.connect(client_uri) as client_ws:
        # Ignore initial empty catchup if any
        try:
            initial = await asyncio.wait_for(client_ws.recv(), timeout=1.0)
            print(f"Initial client message: {initial[:100]}")
        except asyncio.TimeoutError:
            pass

        async with websockets.connect(ingest_uri) as ingest_ws:
            print("Connected to ingest and client WebSockets")
            for i, line in enumerate(lines[:15]): # Run 15 chunks
                if not line.strip(): continue
                parts = line.split(":", 1)
                if len(parts) == 2:
                    speaker, text = parts[0].strip(), parts[1].strip()
                else:
                    speaker, text = "UNKNOWN", line.strip()
                    
                chunk_data = {
                    "chunk_id": i + 1,
                    "speaker": speaker,
                    "text": text,
                    "timestamp": 1000 + i * 10
                }
                
                await ingest_ws.send(json.dumps(chunk_data))
                ack_str = await ingest_ws.recv()
                print(f"Ingest ack: {ack_str}")
                await asyncio.sleep(1.5)
                
            # Send meeting end
            await ingest_ws.send(json.dumps({"type": "meeting_end"}))
            ack_end = await ingest_ws.recv()
            print(f"End ack: {ack_end}")
            
        print("Finished sending. Collecting client outputs...")
        outputs = []
        summaries = []
        try:
            while True:
                msg_str = await asyncio.wait_for(client_ws.recv(), timeout=20.0)
                msg = json.loads(msg_str)
                if msg.get("type") == "ping":
                    continue
                if msg.get("type") == "meeting_end_broadcast":
                    print("Meeting end broadcast received by client")
                    break
                outputs.append(msg)
                if msg.get("type") == "agent_output" and msg.get("agent_type") == "rolling_summary":
                    summaries.append(msg)
        except asyncio.TimeoutError:
            print("Timed out waiting for client messages")
            
    print(f"\n--- Test Results ---")
    print(f"Total outputs received (chunks + agents): {len(outputs)}")
    
    agent_outputs = [m for m in outputs if m.get("type") == "agent_output"]
    print(f"Total agent outputs: {len(agent_outputs)}")
    
    fc_citations_valid = True
    for msg in agent_outputs:
        agent = msg.get("agent_type")
        content = msg.get("content", "")
        citations = msg.get("citations", [])
        
        if agent in ["fact_check", "context_rag"]:
            print(f"[{agent}] Chunk {msg.get('chunk_id')}: Citations: {citations}")
            if content != "unverifiable" and not citations:
                print(f"  -> WARNING: Missing citations for {agent}")
                fc_citations_valid = False
            elif citations:
                for cit in citations:
                    if ".txt" not in cit:
                        print(f"  -> WARNING: Citation {cit} might not point to a source doc")
                        fc_citations_valid = False
                        
    print(f"\nSummary Updates:")
    for s in summaries:
        print(f"Chunk {s.get('chunk_id')}: {s.get('content')[:150]}...")
        
    if len(summaries) >= 2:
        if summaries[0]['content'] == summaries[1]['content']:
            print("WARNING: Summary at N-1 is identical to N. Incremental update might be failing.")
        else:
            print("Summary appears to be updating incrementally.")

if __name__ == "__main__":
    asyncio.run(test_correctness())
