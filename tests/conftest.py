"""Shared fixtures for the test suite.

Session-scoped: loading the embedding model and Qwen2.5-1.5B-Instruct is
the expensive part of these tests, not the pipeline logic itself, so each
is loaded once per test run and reused across test modules.

chunk_records/embeddings_data re-run the real Step 1/Step 2 pipeline
functions rather than reading pre-existing files, so the test suite is
self-contained (works from a clean checkout) and exercises ingestion and
embedding as real work, not just as fixtures.
"""

import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.generation.generate import load_llm  # noqa: E402
from src.ingestion import ingest  # noqa: E402
from src.retrieval import embed as embed_module  # noqa: E402
from src.retrieval.search import load_embeddings, load_model  # noqa: E402


@pytest.fixture(scope="session")
def chunk_records():
    """Run the real ingestion stage (Step 1) once for the whole test session."""
    return ingest.run()


@pytest.fixture(scope="session")
def embeddings_data(chunk_records):
    """Run the real embedding stage (Step 2), then load what it wrote."""
    embed_module.run()
    return load_embeddings()


@pytest.fixture(scope="session")
def retrieval_model(embeddings_data):
    """The Step 3 sentence-embedding model, loaded once."""
    return load_model(embeddings_data)


@pytest.fixture(scope="session")
def llm():
    """(tokenizer, model) for Qwen2.5-1.5B-Instruct, from the local HF cache."""
    return load_llm()
