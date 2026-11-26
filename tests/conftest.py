"""
conftest.py — pytest fixtures shared across all test modules.

Prerequisites:
    1. Server must be running: uvicorn backend.main:app --host 0.0.0.0 --port 8000
    2. Knowledge base must be ingested: python -m backend.rag.ingest_docs
"""
from __future__ import annotations
import asyncio
import pytest
import os
import httpx

SERVER_HTTP = os.getenv("LMC_SERVER", "http://localhost:8000")


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "live: mark test as requiring a running server"
    )


@pytest.fixture(scope="session", autouse=True)
def verify_server_running():
    """Fail fast if the server is not up before any test runs."""
    try:
        resp = httpx.get(f"{SERVER_HTTP}/health", timeout=5.0)
        resp.raise_for_status()
    except Exception as exc:
        pytest.exit(
            f"\n\n[FATAL] Server is not running at {SERVER_HTTP}\n"
            f"  Start it with: uvicorn backend.main:app --host 0.0.0.0 --port 8000\n"
            f"  Error: {exc}\n",
            returncode=1,
        )


@pytest.fixture
def event_loop():
    """Provide a single event loop per test function."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


def run(coro):
    """Helper: run a coroutine synchronously in tests that aren't async."""
    return asyncio.get_event_loop().run_until_complete(coro)
