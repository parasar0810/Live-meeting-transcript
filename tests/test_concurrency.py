import asyncio
import json
import uuid
import websockets

from tests.helpers import SERVER_WS

async def test_late_joiner():
    print("\n--- Test: Late Joiner ---")
    session_id = f"test-late-{uuid.uuid4().hex[:8]}"
    ingest_uri = f"{SERVER_WS}/ws/{session_id}/ingest"
    client1_uri = f"{SERVER_WS}/ws/{session_id}/client"
    
    async with websockets.connect(client1_uri) as client1_ws:
        # Client 1 connects early
        async with websockets.connect(ingest_uri) as ingest_ws:
            for i in range(10):
                chunk = {"chunk_id": i+1, "speaker": "TEST", "text": f"Chunk {i+1}", "timestamp": 1000}
                await ingest_ws.send(json.dumps(chunk))
                await ingest_ws.recv()
            await asyncio.sleep(2.0) # Wait for processing
            
        print("10 chunks sent. Connecting late joiner...")
        
        async with websockets.connect(client1_uri) as client2_ws:
            catchup_str = await asyncio.wait_for(client2_ws.recv(), timeout=5.0)
            catchup = json.loads(catchup_str)
            assert catchup.get("type") == "catchup", "Late joiner didn't receive catchup"
            print(f"Late joiner received catchup: {len(catchup.get('transcript', []))} chunks, {len(catchup.get('agent_outputs', []))} outputs")
            assert len(catchup.get('transcript', [])) == 10, "Catchup didn't have 10 chunks"

async def test_rapid_fire():
    print("\n--- Test: Rapid Fire ---")
    session_id = f"test-rapid-{uuid.uuid4().hex[:8]}"
    ingest_uri = f"{SERVER_WS}/ws/{session_id}/ingest"
    
    async with websockets.connect(ingest_uri) as ingest_ws:
        for i in range(10):
            chunk = {"chunk_id": i+1, "speaker": "TEST", "text": f"Rapid {i+1}", "timestamp": 1000}
            await ingest_ws.send(json.dumps(chunk))
        
        print("Sent 10 chunks in rapid succession.")
        acks = []
        for _ in range(10):
            acks.append(await ingest_ws.recv())
        print(f"Received {len(acks)} acks successfully.")

async def test_cross_session_leakage():
    print("\n--- Test: Cross-Session Leakage ---")
    session1 = f"session-a-{uuid.uuid4().hex[:8]}"
    session2 = f"session-b-{uuid.uuid4().hex[:8]}"
    
    c1_uri = f"{SERVER_WS}/ws/{session1}/client"
    c2_uri = f"{SERVER_WS}/ws/{session2}/client"
    i1_uri = f"{SERVER_WS}/ws/{session1}/ingest"
    i2_uri = f"{SERVER_WS}/ws/{session2}/ingest"
    
    async with websockets.connect(c1_uri) as c1_ws:
        async with websockets.connect(i2_uri) as i2_ws:
            chunk = {"chunk_id": 1, "speaker": "SESSION B", "text": "Secret session B message", "timestamp": 1000}
            await i2_ws.send(json.dumps(chunk))
            await i2_ws.recv()
            
            try:
                msg = await asyncio.wait_for(c1_ws.recv(), timeout=2.0)
                msg_data = json.loads(msg)
                if msg_data.get("type") != "ping":
                    print(f"FAIL: Session A received data: {msg_data}")
                    assert False, "Data leaked!"
            except asyncio.TimeoutError:
                print("PASS: Session A did not receive Session B's data.")

async def main():
    await test_late_joiner()
    await test_rapid_fire()
    await test_cross_session_leakage()
    print("\nConcurrency tests complete.")

if __name__ == "__main__":
    asyncio.run(main())
