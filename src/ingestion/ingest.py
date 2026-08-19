"""Load raw .txt documents, chunk them, and save the chunks for later embedding.

Pipeline: data/raw/*.txt -> load -> normalize whitespace -> overlapping chunks
          -> data/processed/chunks.json

Standard-library only, no RAG framework.
"""

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = PROCESSED_DIR / "chunks.json"

CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30


def load_documents(raw_dir: Path = RAW_DIR) -> dict[str, str]:
    """Read every .txt file in raw_dir into {filename: raw_text}."""
    documents = {}
    for path in sorted(raw_dir.glob("*.txt")):
        documents[path.name] = path.read_text(encoding="utf-8")
    return documents


def normalize_whitespace(text: str) -> str:
    """Collapse repeated spaces/tabs and excess blank lines; trim ends."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE_WORDS,
    overlap: int = CHUNK_OVERLAP_WORDS,
) -> list[str]:
    """Split text into word-based chunks, each overlapping the previous one."""
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks


def build_chunk_records(documents: dict[str, str]) -> list[dict]:
    """Turn {filename: text} into a flat list of chunk records with metadata."""
    records = []
    for source_document, raw_text in documents.items():
        clean_text = normalize_whitespace(raw_text)
        for chunk_index, chunk in enumerate(chunk_text(clean_text)):
            records.append(
                {
                    "chunk_id": f"{source_document}::{chunk_index}",
                    "source_document": source_document,
                    "chunk_index": chunk_index,
                    "text": chunk,
                }
            )
    return records


def save_chunks(records: list[dict], output_path: Path = OUTPUT_FILE) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def run() -> list[dict]:
    documents = load_documents()
    records = build_chunk_records(documents)
    save_chunks(records)
    return records


if __name__ == "__main__":
    records = run()
    n_docs = len({r["source_document"] for r in records})
    print(f"Loaded {n_docs} document(s) from {RAW_DIR}")
    print(f"Produced {len(records)} chunk(s)")
    print(f"Saved to {OUTPUT_FILE}")
