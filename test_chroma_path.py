import sys
import os
import traceback

# Test with a path WITHOUT spaces
test_path = os.path.join(os.environ.get("TEMP", "C:\\Temp"), "chroma_test_db")
print(f"Testing ChromaDB with path: {test_path}", flush=True)

try:
    import chromadb
    c = chromadb.PersistentClient(path=test_path)
    print("SUCCESS - ChromaDB works with no-space path", flush=True)
except Exception as e:
    print(f"FAIL: {e}", flush=True)
    traceback.print_exc()
