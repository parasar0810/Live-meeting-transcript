"""
ChromaDB wrapper — singleton used by all agents at runtime.

Design decisions:
- Persistent embedded mode: no separate Chroma server process needed for a
  single-node deployment. The DB files live in CHROMA_PATH.
- Singleton pattern: ChromaStore is instantiated once at application startup
  (via lifespan) and shared across all agent calls. ChromaDB's embedded client
  is NOT thread-safe for concurrent writes, but we are read-only at runtime
  (ingestion happens once, before the server starts), so sharing is safe.
- Local embeddings: uses ChromaDB's built-in ONNX MiniLM-L6-V2 model by
  default. This is free (no API calls) and works well for document retrieval.
  The model is small (~80MB) and loads on first use.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import chromadb

from backend.config import settings

logger = logging.getLogger(__name__)

_CHROMA_STORE: Optional["ChromaStore"] = None


class RetrievedDoc:
    """Lightweight container for a single retrieval result."""

    def __init__(self, source: str, content: str, distance: float):
        self.source = source          # document filename / title
        self.content = content        # text of the retrieved chunk
        self.distance = distance      # cosine distance (lower = more relevant)

    def __repr__(self) -> str:
        return f"RetrievedDoc(source={self.source!r}, distance={self.distance:.3f})"


def _get_embedding_function():
    """Return the appropriate embedding function based on config."""
    if settings.embedding_model == "local":
        # Use ChromaDB's built-in ONNX MiniLM-L6-V2 — free, no API calls.
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        logger.info("Using local ONNX MiniLM-L6-V2 embeddings (no API calls)")
        return ONNXMiniLM_L6_V2()
    else:
        # Use OpenAI embeddings (requires valid OpenAI API key with quota).
        from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
        logger.info("Using OpenAI embeddings: %s", settings.embedding_model)
        return OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model,
        )


class ChromaStore:
    """
    Thin wrapper around a ChromaDB persistent collection.

    All reads are synchronous because chromadb's Python client is synchronous.
    Callers in async contexts should use asyncio.to_thread or run_in_executor.
    """

    def __init__(self) -> None:
        chroma_path = str(settings.chroma_path.resolve())
        logger.info("Initialising ChromaDB at %s", chroma_path)

        self._client = chromadb.PersistentClient(path=chroma_path)
        self._embed_fn = _get_embedding_function()
        # get_or_create so the server can start even if ingest hasn't been run yet.
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            embedding_function=self._embed_fn,
            metadata={"hnsw:space": "cosine"},
        )
        doc_count = self._collection.count()
        logger.info(
            "ChromaDB ready — collection '%s' has %d document chunks",
            settings.chroma_collection,
            doc_count,
        )
        if doc_count == 0:
            logger.warning(
                "Knowledge base is empty. Run `python -m backend.rag.ingest_docs` "
                "before starting the server to enable Fact-Check and Context agents."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 3,
        max_distance: float = 1.0,
    ) -> list[RetrievedDoc]:
        """
        Return up to `k` document chunks most similar to `query`.

        Args:
            query:        The search string (a claim or ambiguous reference).
            k:            Maximum number of results.
            max_distance: Filter out results with cosine distance above this
                          threshold. Default 1.0 (no filter) — individual agents
                          can pass a stricter value.

        Returns:
            List of RetrievedDoc, sorted by ascending distance (best first).
            Returns empty list if the collection is empty or no results pass
            the distance filter — never raises.
        """
        if self._collection.count() == 0:
            return []

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=min(k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.error("ChromaDB query failed: %s", exc)
            return []

        docs: list[RetrievedDoc] = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            if dist <= max_distance:
                docs.append(
                    RetrievedDoc(
                        source=meta.get("source", "unknown"),
                        content=doc,
                        distance=dist,
                    )
                )

        return docs

    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# Module-level accessor — lazy singleton
# ---------------------------------------------------------------------------


def get_chroma_store() -> ChromaStore:
    """
    Return the module-level ChromaStore singleton, creating it if necessary.

    In production the FastAPI lifespan handler calls this eagerly at startup.
    During testing it is called on first use.
    """
    global _CHROMA_STORE
    if _CHROMA_STORE is None:
        _CHROMA_STORE = ChromaStore()
    return _CHROMA_STORE
