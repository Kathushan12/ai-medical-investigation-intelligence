# Design Decisions and Trade-offs

## Why FastAPI?

FastAPI is lightweight, fast to develop with, and automatically generates OpenAPI documentation. This is suitable for internship evaluation because the APIs can be tested directly from `/docs`.

## Why PostgreSQL + pgvector?

The assignment requires PostgreSQL and pgvector. This keeps structured metadata, full-text search, and vector search inside one database, which simplifies deployment and improves maintainability.

## Why AI-powered OCR?

Traditional OCR can read characters but often fails with handwriting, complex layouts, and low-quality scans. This system uses multimodal AI document understanding as the primary OCR path and returns warnings/confidence scores.

## Why chunking?

Long reports are split into overlapping chunks so retrieval can find the most relevant evidence. Chunk-level citations are also more precise than whole-document citations.

## Why hybrid search?

Keyword search is good for exact terms such as patient names or hospital names. Semantic search is good for meaning-based queries such as “similar diagnosis pattern”. Hybrid search combines both.

## Hallucination prevention

The assistant receives only retrieved evidence. The prompt instructs it to answer only from that evidence and return `No supporting evidence found.` when the context is not enough.

## Background processing trade-off

This starter uses FastAPI background tasks to keep setup simple for a 7-day internship task. For a larger production system, Celery/RQ with Redis and separate workers should replace this.

## Privacy note

Real medical data should not be committed to GitHub. Use synthetic samples only.
