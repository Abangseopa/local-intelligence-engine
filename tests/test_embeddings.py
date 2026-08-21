"""Boundary checks for Step 2: local embeddings (src/retrieval/embed.py)."""


def test_embeddings_have_one_vector_per_chunk_at_correct_dimension(chunk_records, embeddings_data):
    assert embeddings_data["dimension"] == 384  # all-MiniLM-L6-v2's output size
    assert len(embeddings_data["embeddings"]) == len(chunk_records)


def test_each_embedding_traces_back_to_its_source_chunk(chunk_records, embeddings_data):
    chunk_ids = {record["chunk_id"] for record in chunk_records}
    for entry in embeddings_data["embeddings"]:
        assert entry["chunk_id"] in chunk_ids
        assert len(entry["embedding"]) == embeddings_data["dimension"]
