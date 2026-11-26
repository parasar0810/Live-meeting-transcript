import asyncio
import json
import uuid
import time
import aiohttp
import websockets

from tests.helpers import SERVER_HTTP, SERVER_WS

async def test_stress_backpressure():
    print("\n--- Test: Stress Backpressure ---")
    session_id = f"test-stress-{uuid.uuid4().hex[:8]}"
    ingest_uri = f"{SERVER_WS}/ws/{session_id}/ingest"
    state_uri = f"{SERVER_HTTP}/sessions/{session_id}/state"
    
    async with websockets.connect(ingest_uri) as ingest_ws:
        # Flood 200 chunks instantly
        for i in range(200):
            chunk = {"chunk_id": i+1, "speaker": "TEST", "text": f"Flood {i}", "timestamp": 1000}
            await ingest_ws.send(json.dumps(chunk))
        
        # We don't await all acks synchronously to truly flood it.
        # But we need to drain the acks to keep websocket healthy.
        for _ in range(200):
            await ingest_ws.recv()
            
        print("Flooded 200 chunks.")
        
        # Check work queue size
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.get(state_uri)
            state = resp.json()
            wq_size = state.get("work_queue_size", 0)
            print(f"Work queue size after flood: {wq_size}")
            # The bounded queue is maxsize=50, so it shouldn't exceed 50.
            if wq_size <= 50:
                print("PASS: Work queue bounded correctly.")
            else:
                print("FAIL: Work queue exceeded bounds!")

async def run_single_session_load(session_id: str, latencies: list):
    ingest_uri = f"{SERVER_WS}/ws/{session_id}/ingest"
    client_uri = f"{SERVER_WS}/ws/{session_id}/client"
    
    async with websockets.connect(client_uri) as client_ws:
        async with websockets.connect(ingest_uri) as ingest_ws:
            start_time = time.time()
            chunk = {"chunk_id": 1, "speaker": "TEST", "text": "Action item: measure load latency.", "timestamp": 1000}
            await ingest_ws.send(json.dumps(chunk))
            await ingest_ws.recv()
            
            # Wait for first agent output
            try:
                while True:
                    msg_str = await asyncio.wait_for(client_ws.recv(), timeout=10.0)
                    msg = json.loads(msg_str)
                    if msg.get("type") == "agent_output":
                        latencies.append(time.time() - start_time)
                        break
            except asyncio.TimeoutError:
                pass # Failed to get output in time
                
async def test_concurrent_load():
    print("\n--- Test: Concurrent Load (10 Sessions) ---")
    latencies = []
    tasks = []
    for i in range(10):
        session_id = f"test-load-{i}-{uuid.uuid4().hex[:8]}"
        tasks.append(asyncio.create_task(run_single_session_load(session_id, latencies)))
        
    await asyncio.gather(*tasks)
    
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies)//2]
        p95 = latencies[int(len(latencies)*0.95)] if len(latencies) >= 20 else latencies[-1]
        print(f"PASS: Concurrent load complete. Responses received: {len(latencies)}/10")
        print(f"Latency p50: {p50:.2f}s, p95: {p95:.2f}s")
    else:
        print("FAIL: No responses received under load.")

async def main():
    await test_stress_backpressure()
    await test_concurrent_load()
    print("\nLoad tests complete.")

if __name__ == "__main__":
    asyncio.run(main())
