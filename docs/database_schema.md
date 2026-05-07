# Database Schema

## investigations

Stores one uploaded investigation report/evidence document.

| Column | Purpose |
|---|---|
| id | UUID primary key |
| original_filename | Uploaded file name |
| stored_filename | Safe stored file name |
| file_path | Server path of evidence file |
| mime_type | File type |
| upload_status | uploaded / failed |
| ocr_status | pending / processing / completed / failed |
| embedding_status | pending / processing / completed / failed |
| processing_status | pending / processing / completed / failed |
| extracted_text | Full AI OCR text |
| summary | Extracted summary |
| patient_name | Extracted patient name |
| medical_condition | Extracted condition/diagnosis |
| incident_date | Extracted date |
| hospital_location | Extracted hospital/location |
| severity_level | Low / Medium / High / Critical / Unknown |
| doctor_notes | Extracted doctor notes |
| lab_test_details | Extracted lab/test evidence |
| ocr_confidence | Average OCR confidence |
| extraction_metadata | JSON warnings/page metadata |
| error_message | Processing error message |
| created_at / updated_at | Audit fields |

## evidence_chunks

Stores searchable chunks from investigation evidence.

| Column | Purpose |
|---|---|
| id | UUID primary key |
| investigation_id | Foreign key to investigations |
| chunk_index | Position of chunk in document |
| content | Chunk text |
| source_page | Source page if known |
| confidence | Confidence score inherited from OCR |
| embedding | pgvector embedding |
| created_at | Audit field |

## Indexes

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE INDEX ix_chunks_embedding_hnsw ON evidence_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ix_chunks_fts ON evidence_chunks USING gin (to_tsvector('english', content));
```
