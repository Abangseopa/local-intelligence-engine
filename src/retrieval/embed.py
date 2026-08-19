"""Turn each chunk's text into a numeric embedding vector using a local model.

Pipeline: data/processed/chunks.json -> embedding model -> data/processed/embeddings.json

Uses sentence-transformers to run the model entirely on-device (CPU is fine).
No vector database, no similarity search — just the raw embedding step.
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.json"
EMBEDDINGS_FILE = BASE_DIR / "data" / "processed" / "embeddings.json"

# all-MiniLM-L6-v2: ~80MB, 384-dim, runs comfortably on CPU, strong general-purpose
# semantic similarity performance for its size. See README for rationale.
MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(chunks_file: Path = CHUNKS_FILE) -> list[dict]:
    """Read the chunk records produced by the ingestion stage."""
    return json.loads(chunks_file.read_text(encoding="utf-8"))


def embed_chunks(chunks: list[dict], model_name: str = MODEL_NAME) -> list[dict]:
    """Encode each chunk's text into a vector, keeping all original metadata."""
    model = SentenceTransformer(model_name)
    texts = [chunk["text"] for chunk in chunks]
    vectors = model.encode(texts, show_progress_bar=False)

    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append(
            {
                "chunk_id": chunk["chunk_id"],
                "source_document": chunk["source_document"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "embedding": vector.tolist(),
            }
        )
    return records


def save_embeddings(
    records: list[dict],
    model_name: str,
    output_path: Path = EMBEDDINGS_FILE,
) -> None:
    """Write embeddings + metadata + the model name that produced them to disk."""
    dimension = len(records[0]["embedding"]) if records else 0
    payload = {
        "model": model_name,
        "dimension": dimension,
        "embeddings": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run() -> dict:
    chunks = load_chunks()
    records = embed_chunks(chunks)
    save_embeddings(records, MODEL_NAME)
    dimension = len(records[0]["embedding"]) if records else 0
    return {"model": MODEL_NAME, "dimension": dimension, "count": len(records)}


if __name__ == "__main__":
    summary = run()
    print(f"Embedded {summary['count']} chunk(s) using {summary['model']}")
    print(f"Embedding dimension: {summary['dimension']}")
    print(f"Saved to {EMBEDDINGS_FILE}")
