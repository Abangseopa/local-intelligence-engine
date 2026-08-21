"""Boundary checks for Step 1: ingestion and chunking (src/ingestion/ingest.py)."""

from src.ingestion.ingest import chunk_text, normalize_whitespace


def test_run_produces_chunks_with_consistent_metadata(chunk_records):
    assert len(chunk_records) > 0
    for record in chunk_records:
        assert record["chunk_id"] == f"{record['source_document']}::{record['chunk_index']}"
        assert record["text"].strip() != ""


def test_chunk_text_overlap_boundary_repeats_words_verbatim():
    # 160 words with chunk_size=150, overlap=30 (step=120) lands exactly on
    # the boundary: two chunks, sharing the 30-word overlap region.
    words = [f"word{i}" for i in range(160)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=150, overlap=30)

    assert len(chunks) == 2
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    assert len(first_words) == 150
    assert first_words[-30:] == second_words[:30]


def test_chunk_text_empty_input_produces_no_chunks():
    assert chunk_text("") == []


def test_normalize_whitespace_collapses_spaces_and_blank_lines():
    messy = "a   b\n\n\n\nc\t\td"
    assert normalize_whitespace(messy) == "a b\n\nc d"
