"""
One-time knowledge base ingestion script.

Usage:
    python -m backend.rag.ingest_docs

This script reads all .txt files from knowledge_base/sample_docs/, splits
them into overlapping chunks, and stores them in the ChromaDB persistent
collection defined in config.py.

Run this ONCE before starting the server. Re-running it will upsert
(update existing + add new) docs, so it is idempotent.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb

from backend.config import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

DOCS_DIR = PROJECT_ROOT / "knowledge_base" / "sample_docs"
CHUNK_SIZE = 400        # characters per chunk
CHUNK_OVERLAP = 80      # character overlap between adjacent chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping character-level windows."""
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 20]  # drop tiny trailing fragments


def doc_id(source: str, chunk_index: int) -> str:
    """Stable, deterministic ID for a document chunk."""
    raw = f"{source}::{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _get_embedding_function():
    """Return the appropriate embedding function based on config."""
    if settings.embedding_model == "local":
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        logger.info("Using local ONNX MiniLM-L6-V2 embeddings")
        return ONNXMiniLM_L6_V2()
    else:
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        logger.info("Using OpenAI embeddings: %s", settings.embedding_model)
        return OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model,
        )


# ---------------------------------------------------------------------------
# Main ingestion routine
# ---------------------------------------------------------------------------


def ingest() -> None:
    if not DOCS_DIR.exists():
        logger.error("Docs directory not found: %s", DOCS_DIR)
        sys.exit(1)

    txt_files = sorted(DOCS_DIR.glob("*.txt"))
    if not txt_files:
        logger.error("No .txt files found in %s", DOCS_DIR)
        sys.exit(1)

    logger.info("Found %d documents to ingest", len(txt_files))

    # Build ChromaDB client + embedding function directly here (don't use the
    # singleton, which is designed for the long-running server process).
    chroma_path = str(settings.chroma_path.resolve())
    client = chromadb.PersistentClient(path=chroma_path)
    embed_fn = _get_embedding_function()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0
    for txt_path in txt_files:
        source_name = txt_path.stem  # filename without extension
        raw_text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_text(raw_text, CHUNK_SIZE, CHUNK_OVERLAP)
        logger.info("  %s → %d chunks", txt_path.name, len(chunks))

        ids = [doc_id(source_name, i) for i in range(len(chunks))]
        metadatas = [{"source": source_name, "chunk_index": i} for i in range(len(chunks))]

        # Upsert in batches of 100 (ChromaDB recommended batch size).
        batch_size = 100
        for batch_start in range(0, len(chunks), batch_size):
            batch_end = batch_start + batch_size
            collection.upsert(
                ids=ids[batch_start:batch_end],
                documents=chunks[batch_start:batch_end],
                metadatas=metadatas[batch_start:batch_end],
            )

        total_chunks += len(chunks)

    final_count = collection.count()
    logger.info(
        "Ingestion complete. %d chunks added/updated. Collection now has %d total chunks.",
        total_chunks,
        final_count,
    )


if __name__ == "__main__":
    ingest()
