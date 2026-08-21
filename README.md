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

### 3. Semantic retrieval (`src/retrieval/search.py`)

Ranks stored chunks by how semantically similar they are to a query:

```
query text → same embedding model as Step 3 → query vector
           → cosine similarity vs. every stored chunk vector (NumPy)
           → ranked results → top-k
```

- Loads `data/processed/embeddings.json` and reads the model name stored
  inside it, so the query is always embedded with the exact same model used
  for the chunks (required for the vectors to be comparable at all).
- Embeds the query, then computes cosine similarity between the query vector
  and every chunk vector directly with NumPy (`(a · b) / (|a| |b|)`) — no
  FAISS or vector database yet, since the dataset is small enough to compare
  against every vector and see the math explicitly.
- Sorts chunks by similarity score (descending) and returns the top-k, each
  with `rank`, `similarity_score`, `chunk_id`, `source_document`,
  `chunk_index`, and `text`.

Run the demo from the project root (after running ingestion and embedding):

```
python3 src/retrieval/search.py
```

The demo runs five queries against the example mining report, including one
paraphrased question with little keyword overlap with the source text, to
show that retrieval works on meaning rather than exact word matches.

### 4. Grounded generation (`src/generation/generate.py`)

Turns retrieved chunks into a grounded natural-language answer using a local,
open-source instruction-tuned LLM:

```
question → Step 3 retrieval (top-k chunks) → build_prompt(question, chunks)
         → Qwen2.5-1.5B-Instruct (local) → answer
```

- Reuses `src.retrieval.search` directly for retrieval — no retrieval logic
  is duplicated here. This module only adds prompt construction and LLM
  inference on top of it.
- Builds a single prompt that instructs the model to answer using ONLY the
  retrieved CONTEXT, and to say clearly that it doesn't know rather than
  guess when the context doesn't contain the answer.
- Generates the answer locally with `Qwen/Qwen2.5-1.5B-Instruct` (via
  `transformers`), using greedy decoding for deterministic output. Runs on
  Apple Silicon MPS if available, otherwise CPU.

**Why `Qwen2.5-1.5B-Instruct`:** it's a small (~1.5B parameter, ~3GB)
instruction-tuned model released under Apache-2.0, ungated on Hugging Face,
with strong instruction-following for its size, and runs comfortably on an
Apple Silicon Mac (CPU or MPS) without a rented GPU.

Run the demo from the project root (after running ingestion, embedding, and
retrieval are available, i.e. `data/processed/embeddings.json` exists):

```
python3 src/generation/generate.py
```

The demo runs four questions against the example mining report: two directly
answerable from the retrieved evidence, one paraphrased question, and one
whose answer is deliberately absent from the source document — to confirm
the model declines to answer rather than hallucinating when the context
doesn't support an answer.

Requires `torch` and `transformers` (see `requirements.txt`); the model
weights are downloaded once from Hugging Face and cached locally
(`~/.cache/huggingface/hub`) on first run.

## Status

Ingestion, chunking, local embeddings, semantic retrieval, and grounded
generation implemented (Step 5/8). Evaluation is not yet implemented.
