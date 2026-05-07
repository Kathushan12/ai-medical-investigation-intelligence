# API Documentation

Backend base URL: `http://localhost:8000/api/v1`

## Investigations

### Upload evidence

`POST /investigations/upload`

Form-data:

- `file`: PDF, PNG, JPG, JPEG, TXT, or scanned evidence file.

Returns investigation record and starts background processing.

### List investigations

`GET /investigations`

### Get one investigation

`GET /investigations/{investigation_id}`

### Get processing status

`GET /investigations/{investigation_id}/status`

### Reprocess

`POST /investigations/{investigation_id}/reprocess`

### Related investigations

`GET /investigations/{investigation_id}/related?limit=5`

Uses extracted fields and semantic search to find similar reports.

## Search

### Keyword search

`GET /search/keyword?q=diabetes&limit=10`

Uses PostgreSQL full-text search.

### Semantic search

`GET /search/semantic?q=high risk diabetes in Colombo&limit=10`

Uses embeddings and pgvector cosine similarity.

### Hybrid search

`GET /search/hybrid?q=fraud related investigations&limit=10`

Combines keyword and semantic results.

Optional filters:

- `severity_level`
- `hospital_location`
- `medical_condition`

## Assistant

### Ask grounded question

`POST /assistant/query`

```json
{
  "question": "Show high-risk incidents in Colombo",
  "top_k": 5,
  "severity_level": "High",
  "hospital_location": "Colombo"
}
```

The assistant returns an answer, citations, and retrieved evidence. If no evidence exists, it returns `No supporting evidence found.`
