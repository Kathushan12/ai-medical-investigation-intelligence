# AI-Powered Medical Investigation Report Intelligence System

This repository is a production-oriented starter implementation for the **AI Internship Task Assignment**.

## What this system does

- Uploads investigation evidence: images, scanned documents, PDFs, and text samples.
- Uses AI-powered document understanding for OCR instead of relying only on traditional OCR.
- Extracts structured fields such as patient name, condition, hospital/location, severity, doctor notes, and lab/test details.
- Chunks investigation text and stores embeddings in PostgreSQL with pgvector.
- Supports keyword search, semantic search, and hybrid search.
- Provides a grounded AI assistant that answers only from retrieved evidence and returns citations.
- Tracks upload, OCR, embedding, and processing statuses.
- Includes Docker, API documentation, database schema, architecture notes, sample files, and daily report templates.

## Tech stack

Backend: FastAPI, SQLAlchemy, PostgreSQL, pgvector, OpenAI API, pypdfium2, Pillow  
Frontend: Next.js, React, TypeScript, Tailwind CSS  
Deployment: Docker and Docker Compose

## Repository structure

```text
.
├── backend/
│   ├── app/
│   │   ├── api/routes/          # FastAPI endpoints
│   │   ├── services/            # OCR, extraction, embedding, RAG, search
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── main.py
│   │   ├── models.py
│   │   └── schemas.py
│   ├── storage/uploads/
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
├── docs/
├── samples/
├── docker-compose.yml
└── .env.example
```

## Day 1 setup

```bash
git init
git add .
git commit -m "day 1: initialize medical investigation intelligence system"

cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

docker compose up --build
```

Open:

- Frontend: http://localhost:3000
- Backend health: http://localhost:8000/health
- API docs: http://localhost:8000/docs

## Useful API commands

Upload a sample report:

```bash
curl -X POST "http://localhost:8000/api/v1/investigations/upload" \
  -F "file=@samples/sample_lab_report.txt"
```

List investigations:

```bash
curl "http://localhost:8000/api/v1/investigations"
```

Hybrid search:

```bash
curl "http://localhost:8000/api/v1/search/hybrid?q=high-risk%20diabetes%20Colombo&limit=5"
```

Ask the AI assistant:

```bash
curl -X POST "http://localhost:8000/api/v1/assistant/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"Show high-risk incidents in Colombo","top_k":5}'
```

## Seven-day GitHub plan

### Day 1 - Project setup and architecture

- Create GitHub repository.
- Add Docker Compose, FastAPI skeleton, PostgreSQL + pgvector.
- Add architecture, schema, and design decision docs.
- Push commit:

```bash
git add .
git commit -m "day 1: setup architecture fastapi postgres pgvector docker"
git push origin main
```

### Day 2 - Upload and investigation management

- Implement upload API.
- Save evidence files safely.
- Create investigation records.
- Add status tracking fields.
- Push commit:

```bash
git add .
git commit -m "day 2: add evidence upload and investigation management apis"
git push origin main
```

### Day 3 - AI OCR and structured extraction

- Add PDF-to-image conversion.
- Add AI vision OCR.
- Add structured extraction JSON schema.
- Store extracted text and metadata.
- Push commit:

```bash
git add .
git commit -m "day 3: implement ai ocr and structured medical extraction"
git push origin main
```

### Day 4 - Chunking, embeddings, and pgvector storage

- Preprocess and chunk extracted text.
- Generate embeddings.
- Store chunks in PostgreSQL vector columns.
- Add vector index.
- Push commit:

```bash
git add .
git commit -m "day 4: add chunking embeddings and pgvector persistence"
git push origin main
```

### Day 5 - Search and related investigation detection

- Add PostgreSQL full-text keyword search.
- Add semantic vector search.
- Add hybrid search.
- Add related investigation detection.
- Push commit:

```bash
git add .
git commit -m "day 5: implement keyword semantic hybrid search and related detection"
git push origin main
```

### Day 6 - RAG assistant and frontend

- Add grounded AI assistant with citations.
- Add hallucination prevention fallback.
- Build upload/search/assistant UI.
- Push commit:

```bash
git add .
git commit -m "day 6: add rag assistant citations and frontend ui"
git push origin main
```

### Day 7 - Testing, documentation, and final cleanup

- Add tests.
- Update README.
- Add API documentation, DB schema, architecture diagram, design decisions, and day report.
- Final push:

```bash
git add .
git commit -m "day 7: finalize documentation tests and deployment setup"
git push origin main
```

## Important notes for evaluation

- Do not submit only screenshots. The repository must run with Docker.
- Keep daily commits visible on GitHub.
- Add screenshots or short screen recordings only as extra proof, not as the main deliverable.
- Use sample files only for testing. Do not upload real patient data.
