"""Rank saved chunk embeddings by cosine similarity to a query.

Pipeline: query text -> same embedding model used in Step 3 -> query vector
          -> cosine similarity against every chunk vector (NumPy, explicit)
          -> sorted ranking -> top-k results

No FAISS or vector database yet — the dataset is small enough to compare
the query against every stored vector directly, so the similarity math is
fully visible.
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
EMBEDDINGS_FILE = BASE_DIR / "data" / "processed" / "embeddings.json"

DEFAULT_TOP_K = 3


def load_embeddings(embeddings_file: Path = EMBEDDINGS_FILE) -> dict:
    """Load the {model, dimension, embeddings} payload written by embed.py."""
    return json.loads(embeddings_file.read_text(encoding="utf-8"))


def load_model(data: dict) -> SentenceTransformer:
    """Load the exact model that produced these embeddings, by name."""
    return SentenceTransformer(data["model"])


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """Embed the query text with the same model used for the chunks."""
    return np.array(model.encode(query))


def cosine_similarity(query_vector: np.ndarray, chunk_matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity between one query vector and a matrix of chunk vectors.

    cos(theta) = (a . b) / (|a| * |b|)

    Result is in [-1, 1]; 1 means the vectors point in the same direction
    (most similar meaning), 0 means unrelated, -1 means opposite.
    """
    query_unit = query_vector / np.linalg.norm(query_vector)
    chunk_norms = np.linalg.norm(chunk_matrix, axis=1, keepdims=True)
    chunk_units = chunk_matrix / chunk_norms
    return chunk_units @ query_unit


def search(
    query: str,
    model: SentenceTransformer,
    data: dict,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    """Embed `query`, score it against every stored chunk, return top_k ranked results."""
    embeddings = data["embeddings"]
    chunk_matrix = np.array([e["embedding"] for e in embeddings])

    query_vector = embed_query(query, model)
    scores = cosine_similarity(query_vector, chunk_matrix)

    ranked_indices = np.argsort(-scores)[:top_k]

    results = []
    for rank, idx in enumerate(ranked_indices, start=1):
        chunk = embeddings[idx]
        results.append(
            {
                "rank": rank,
                "similarity_score": float(scores[idx]),
                "chunk_id": chunk["chunk_id"],
                "source_document": chunk["source_document"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
            }
        )
    return results


# Demo queries target distinct facts in the fictional mining report.
# The last one is a deliberate paraphrase with little keyword overlap with
# the source text, to demonstrate that retrieval is semantic, not lexical.
DEMO_QUERIES = [
    "What mineral is being mined?",
    "How is the ore processed?",
    "What concentrate grade is produced?",
    "What equipment or mining method is used?",
    "Is the site dangerous for the people who work there?",  # paraphrase, no exact keyword match
]


def print_results(query: str, results: list[dict]) -> None:
    print(f"\nQuery: {query}")
    for r in results:
        excerpt = r["text"][:160].replace("\n", " ") + "..."
        print(
            f"  rank={r['rank']} score={r['similarity_score']:.4f} "
            f"chunk_id={r['chunk_id']} "
            f"(source={r['source_document']}, chunk_index={r['chunk_index']})"
        )
        print(f"    excerpt: {excerpt}")


def run_demo(top_k: int = DEFAULT_TOP_K) -> None:
    data = load_embeddings()
    model = load_model(data)
    for query in DEMO_QUERIES:
        results = search(query, model, data, top_k=top_k)
        print_results(query, results)


if __name__ == "__main__":
    run_demo()
