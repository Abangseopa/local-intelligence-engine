"""Boundary checks for Step 4: semantic retrieval (src/retrieval/search.py)."""

from src.retrieval.search import DEFAULT_TOP_K, search


def test_search_returns_top_k_ranked_by_descending_similarity(embeddings_data, retrieval_model):
    results = search("What mineral is being mined?", retrieval_model, embeddings_data, top_k=3)
    assert len(results) == 3
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_search_finds_expected_chunk_for_direct_question(embeddings_data, retrieval_model):
    results = search(
        "What concentrate grade is produced?", retrieval_model, embeddings_data, top_k=DEFAULT_TOP_K
    )
    retrieved_ids = {r["chunk_id"] for r in results}
    assert "kestrel_ridge_report.txt::1" in retrieved_ids


def test_search_finds_expected_chunk_for_paraphrased_question(embeddings_data, retrieval_model):
    # No keyword overlap with "LTIFR" / "safety induction" in the source
    # text -- a correct match here demonstrates semantic, not lexical, search.
    results = search(
        "Is the site dangerous for the people who work there?",
        retrieval_model,
        embeddings_data,
        top_k=DEFAULT_TOP_K,
    )
    retrieved_ids = {r["chunk_id"] for r in results}
    assert retrieved_ids & {"kestrel_ridge_report.txt::2", "kestrel_ridge_report.txt::3"}


def test_search_top_k_boundary_never_exceeds_available_chunks(embeddings_data, retrieval_model):
    total_chunks = len(embeddings_data["embeddings"])
    results = search("lithium", retrieval_model, embeddings_data, top_k=total_chunks + 10)
    assert len(results) == total_chunks
