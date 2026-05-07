# AI-Powered Medical Investigation Report Intelligence System

A production-oriented AI internship project for building a **Medical Investigation Report Intelligence System** using AI OCR, structured extraction, embeddings, PostgreSQL + pgvector, hybrid search, and a Retrieval-Augmented Generation (RAG) assistant.

This system is designed to process medical investigation evidence such as scanned reports, images, PDFs, and typed documents, extract meaningful medical information, store searchable investigation records, and answer investigation questions using retrieved evidence with citations.

---

## Project Objective

The objective of this project is to demonstrate practical knowledge in:

- AI-powered OCR
- Medical document understanding
- Structured data extraction
- Embeddings and semantic search
- PostgreSQL + pgvector
- Retrieval-Augmented Generation
- FastAPI backend architecture
- Database design
- Background processing
- AI workflow automation
- Clean architecture
- Scalable system design
- Docker-based deployment
- Professional documentation and GitHub workflow

The main focus of this project is not only UI development, but also:

- AI reasoning
- Retrieval quality
- System architecture
- Engineering maturity
- Real-world AI problem solving

---

## What This System Does

The system supports the following complete workflow:

1. Upload investigation evidence.
2. Store the uploaded file safely.
3. Create an investigation record in PostgreSQL.
4. Run AI-powered OCR in the background.
5. Extract structured medical investigation fields.
6. Preprocess and chunk extracted text.
7. Generate embeddings for each text chunk.
8. Store embeddings in PostgreSQL using pgvector.
9. Support keyword search using PostgreSQL full-text search.
10. Support semantic search using vector similarity.
11. Support hybrid search by combining keyword and semantic relevance.
12. Provide an AI assistant using RAG.
13. Return grounded AI answers with citations.
14. Track upload, OCR, embedding, and processing statuses.

---

## Supported Evidence Types

The system supports uploading:

- PDF files
- Scanned medical documents
- PNG images
- JPG/JPEG images
- Typed text reports

Supported file examples:

```text
.pdf
.png
.jpg
.jpeg
.txt
```

## Key Features

### AI-Powered OCR

The system uses AI-based document understanding instead of relying only on traditional OCR. This helps process:

- Scanned reports
- Low-quality images
- Handwritten notes
- Mixed document layouts
- Medical investigation documents
- Tables and notes inside reports

### Structured Data Extraction

The system automatically extracts important fields such as:

- Patient name
- Medical condition
- Incident date
- Hospital or location
- Severity level
- Doctor notes
- Lab/test details
- Summary
- Confidence score
- Warnings

### Background Processing

OCR and embedding generation run in the background. The system tracks:

- Upload status
- OCR status
- Embedding status
- Processing status

## High-Level Architecture

```
User / Investigator
        |
        v
Frontend - Next.js, React, TypeScript, Tailwind CSS
        |
        v
Backend API - FastAPI
        |
        |-----------------------------|
        |                             |
        v                             v
Investigation APIs              AI Assistant APIs
Upload / Status / List          RAG Question Answering
        |
        v
Background Processing Pipeline
        |
        |--> AI OCR
        |--> Structured Extraction
        |--> Text Chunking
        |--> Embedding Generation
        |--> Store in PostgreSQL + pgvector
        |
        v
Retrieval System
        |
        |--> Keyword Search
        |--> Semantic Search
        |--> Hybrid Search
        |
        v
Grounded AI Answer with Citations
```
