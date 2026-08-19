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

## Status

Project scaffolding only. Pipeline stages are not yet implemented.
