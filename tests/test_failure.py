import asyncio
import json
import uuid
import websockets

async def test_malformed_input():
    print("\n--- Test: Malformed Input ---")
    session_id = f"test-fail-{uuid.uuid4().hex[:8]}"
    ingest_uri = f"ws://localhost:8000/ws/{session_id}/ingest"
    
    async with websockets.connect(ingest_uri) as ingest_ws:
        # Invalid JSON
        await ingest_ws.send("{invalid_json:")
        resp = await ingest_ws.recv()
        print(f"Invalid JSON response: {resp}")
        assert "invalid_json" in resp
        
        # Missing required field
        await ingest_ws.send(json.dumps({"chunk_id": 1, "speaker": "TEST"})) # missing text
        resp = await ingest_ws.recv()
        print(f"Missing field response: {resp}")
        assert "validation_error" in resp
        
        # Out of order
        await ingest_ws.send(json.dumps({"chunk_id": 1, "speaker": "T", "text": "A", "timestamp": 1}))
        await ingest_ws.recv() # ack
        await ingest_ws.send(json.dumps({"chunk_id": 3, "speaker": "T", "text": "B", "timestamp": 2}))
        resp = await ingest_ws.recv()
        print(f"Out of order response: {resp}")
        assert "out_of_order_chunk" in resp
        
        print("Session remained open and responded with errors as expected.")

async def test_llm_failure():
    print("\n--- Test: LLM Failure (Token Limit) ---")
    session_id = f"test-llm-{uuid.uuid4().hex[:8]}"
    ingest_uri = f"ws://localhost:8000/ws/{session_id}/ingest"
    
    async with websockets.connect(ingest_uri) as ingest_ws:
        massive_text = "I'll take the action item to " + ("do " * 100000) # Should trigger context limit on gpt-4o-mini
        chunk = {"chunk_id": 1, "speaker": "TEST", "text": massive_text, "timestamp": 1000}
        await ingest_ws.send(json.dumps(chunk))
        ack = await ingest_ws.recv()
        print(f"Ack for massive chunk: {ack}")
        
        # Ensure session is still alive
        chunk2 = {"chunk_id": 2, "speaker": "TEST", "text": "Normal text", "timestamp": 1001}
        await ingest_ws.send(json.dumps(chunk2))
        ack2 = await ingest_ws.recv()
        print(f"Ack for subsequent normal chunk: {ack2}")
        print("Session survived LLM failure.")

async def main():
    await test_malformed_input()
    await test_llm_failure()
    print("\nFailure tests complete.")

if __name__ == "__main__":
    asyncio.run(main())
