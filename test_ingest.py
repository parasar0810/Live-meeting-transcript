import sys
print("Python OK", flush=True)
try:
    print("Importing chromadb...", flush=True)
    import chromadb
    print("chromadb OK", flush=True)
except Exception as e:
    print(f"chromadb FAIL: {e}", flush=True)
    sys.exit(1)

try:
    print("Importing config...", flush=True)
    sys.path.insert(0, ".")
    from backend.config import settings
    print(f"config OK, key={settings.openai_api_key[:10]}...", flush=True)
except Exception as e:
    print(f"config FAIL: {e}", flush=True)
    sys.exit(1)

try:
    print("Creating ChromaDB client...", flush=True)
    chroma_path = str(settings.chroma_path.resolve())
    print(f"  path: {chroma_path}", flush=True)
    client = chromadb.PersistentClient(path=chroma_path)
    print("client OK", flush=True)
except Exception as e:
    print(f"client FAIL: {e}", flush=True)
    sys.exit(1)

try:
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
    print("Creating embed fn...", flush=True)
    embed_fn = OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name=settings.embedding_model,
    )
    print("embed fn OK", flush=True)
except Exception as e:
    print(f"embed fn FAIL: {e}", flush=True)
    sys.exit(1)

try:
    print("Getting collection...", flush=True)
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"collection OK, count={collection.count()}", flush=True)
except Exception as e:
    print(f"collection FAIL: {e}", flush=True)
    sys.exit(1)

try:
    from pathlib import Path
    docs_dir = Path("knowledge_base/sample_docs")
    txt_files = sorted(docs_dir.glob("*.txt"))
    print(f"Found {len(txt_files)} files", flush=True)
    first = txt_files[0]
    text = first.read_text(encoding="utf-8")[:400].strip()
    print(f"Upserting test chunk from {first.name}...", flush=True)
    collection.upsert(
        ids=["test-debug-001"],
        documents=[text],
        metadatas=[{"source": first.stem, "chunk_index": 0}],
    )
    print(f"Upsert OK, count={collection.count()}", flush=True)
except Exception as e:
    print(f"upsert FAIL: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("ALL STEPS PASSED", flush=True)
