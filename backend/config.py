"""
Centralized application configuration.

All settings are read from environment variables (or a .env file at the
workspace root). Defaults are chosen to work for local development without
any additional setup beyond providing an API key.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # LLM Provider (OpenAI-compatible — works with Groq, OpenAI, etc.)
    # ------------------------------------------------------------------
    openai_api_key: str
    """API key for the LLM provider."""

    openai_api_base: Optional[str] = None
    """Base URL for the LLM API. Set to https://api.groq.com/openai/v1
    for Groq, or leave None for OpenAI default."""

    llm_model: str = "llama-3.3-70b-versatile"
    """LLM used by all agents. Default is Groq's llama-3.3-70b."""

    embedding_model: str = "local"
    """Embedding model. Set to 'local' to use ChromaDB's built-in
    ONNX MiniLM-L6-V2 embeddings (free, no API calls). Set to
    'text-embedding-3-small' to use OpenAI embeddings."""

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------
    chroma_path: Path = Path("./knowledge_base/chroma_db")
    """Persistent directory for ChromaDB. Relative to the working directory
    from which the server is launched (i.e. the project root)."""

    chroma_collection: str = "meeting_kb"

    # ------------------------------------------------------------------
    # Supervisor
    # ------------------------------------------------------------------
    supervisor_summary_interval_chunks: int = 5
    """Trigger Rolling Summary every N chunks regardless of content."""

    supervisor_summary_interval_seconds: int = 30
    """Also trigger Rolling Summary if this many seconds have passed since
    the last summary run, even if the chunk threshold hasn't been reached."""

    # ------------------------------------------------------------------
    # Back-pressure
    # ------------------------------------------------------------------
    work_queue_maxsize: int = 50
    """Maximum number of pending agent tasks per session. When exceeded,
    the oldest item is evicted (see supervisor.py for rationale)."""

    client_queue_maxsize: int = 100
    """Maximum number of queued messages per connected browser client.
    Overflow messages for a slow client are dropped silently — we prefer
    losing a stale update over blocking the broadcast path."""

    # ------------------------------------------------------------------
    # Executor
    # ------------------------------------------------------------------
    agent_thread_workers: int = 8
    """Thread-pool size for agent execution. Each agent invocation blocks a
    thread while awaiting LLM I/O; 8 threads allows up to 8 concurrent
    agent runs across all active sessions."""

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = "INFO"


# Module-level singleton — import this everywhere instead of
# constructing a new Settings() per call-site.
settings = Settings()
