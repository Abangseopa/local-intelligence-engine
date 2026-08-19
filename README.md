# Local Intelligence Engine

A local Retrieval-Augmented Generation (RAG) system built end-to-end with
open-source models, created to understand the complete RAG pipeline from
first principles.

## Purpose

This project walks through every stage of a RAG system — ingesting
documents, chunking them, generating embeddings, indexing them locally,
retrieving relevant context, and generating grounded answers with an
open-source language model — with an emphasis on clarity and modularity
over production complexity.

## Architecture

```
documents → chunks → embeddings → local index → retrieval → open model → answer
```

The system is designed to run entirely locally using open-source models —
no external APIs or hosted services are required.

## Project structure

```
data/
    raw/         # Original, unmodified source documents
    processed/   # Cleaned/chunked documents ready for embedding
src/
    ingestion/   # Loading and chunking documents
    retrieval/   # Embedding generation, vector index, semantic search
    generation/  # Open-source LLM integration for grounded answers
    evaluation/  # Pipeline evaluation and metrics
models/          # Local model weights/files
reports/         # Generated evaluation reports and outputs
tests/           # Test suite
```

## Pipeline stages

### 1. Ingestion & chunking (`src/ingestion/ingest.py`)

Turns raw documents into overlapping text chunks ready for embedding:

```
data/raw/*.txt → load → normalize whitespace → overlapping chunks → data/processed/chunks.json
```

- Loads every `.txt` file in `data/raw/`.
- Normalizes basic whitespace (collapses repeated spaces/tabs and excess blank lines).
- Splits each document into overlapping, word-based chunks (150 words per chunk,
  30-word overlap between consecutive chunks).
- Saves the result to `data/processed/chunks.json`, where each chunk record has:
  `chunk_id`, `source_document`, `chunk_index`, `text`.

Run it from the project root:

```
python3 src/ingestion/ingest.py
```

Standard library only — no external dependencies.

### 2. Local embeddings (`src/retrieval/embed.py`)

Turns each chunk's text into a numeric vector using a local, open-source
sentence embedding model:

```
data/processed/chunks.json → embedding model → data/processed/embeddings.json
```

- Loads the chunk records produced by the ingestion stage.
- Encodes each chunk's `text` with `all-MiniLM-L6-v2` (via `sentence-transformers`),
  running entirely on-device — no API calls, no GPU required.
- Saves one 384-dimensional vector per chunk to `data/processed/embeddings.json`,
  alongside the model name and each chunk's `chunk_id`, `source_document`,
  `chunk_index`, and original `text`, so every vector can be traced back to
  its source.

**Why `all-MiniLM-L6-v2`:** it's a small (~80MB), well-established
sentence-transformers model that runs comfortably on CPU with no rented GPU,
downloads quickly, and produces strong general-purpose semantic similarity
results relative to its size — a good balance of speed, size, and quality for
learning the mechanics of embedding-based retrieval.

Run it from the project root (after running the ingestion stage):

```
python3 src/retrieval/embed.py
```

Requires `sentence-transformers` (see `requirements.txt`); no vector database
is used yet — embeddings are stored as plain JSON.

## Status

Ingestion, chunking, and local embeddings implemented (Step 3/8). Vector
indexing, retrieval, generation, and evaluation are not yet implemented.
