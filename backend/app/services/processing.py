import logging
import uuid

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import EvidenceChunk, Investigation
from app.services.chunking import chunk_text
from app.services.embeddings import embedding_service
from app.services.extraction import structured_extraction_service
from app.services.ocr import ai_ocr_service

logger = logging.getLogger(__name__)


def _set_failed(investigation: Investigation, error: Exception) -> None:
    investigation.ocr_status = "failed" if investigation.ocr_status == "processing" else investigation.ocr_status
    investigation.embedding_status = "failed" if investigation.embedding_status == "processing" else investigation.embedding_status
    investigation.processing_status = "failed"
    investigation.error_message = str(error)


def process_investigation_task(investigation_id: str) -> None:
    db = SessionLocal()
    try:
        process_investigation(db, uuid.UUID(investigation_id))
    finally:
        db.close()


def process_investigation(db: Session, investigation_id: uuid.UUID) -> None:
    investigation = db.get(Investigation, investigation_id)
    if not investigation:
        raise ValueError("Investigation not found")

    try:
        investigation.processing_status = "processing"
        investigation.ocr_status = "processing"
        investigation.embedding_status = "pending"
        investigation.error_message = None
        db.commit()

        ocr_result = ai_ocr_service.process_document(investigation.file_path)
        fields = structured_extraction_service.extract(ocr_result)

        investigation.extracted_text = ocr_result.full_text
        investigation.ocr_confidence = fields.get("confidence") or ocr_result.average_confidence
        investigation.patient_name = fields.get("patient_name") or None
        investigation.medical_condition = fields.get("medical_condition") or None
        investigation.incident_date = fields.get("incident_date") or None
        investigation.hospital_location = fields.get("hospital_location") or None
        investigation.severity_level = fields.get("severity_level") or "Unknown"
        investigation.doctor_notes = fields.get("doctor_notes") or None
        investigation.lab_test_details = fields.get("lab_test_details") or None
        investigation.summary = fields.get("summary") or None
        investigation.extraction_metadata = {
            "ocr": ocr_result.metadata,
            "structured_extraction_warnings": fields.get("warnings", []),
            "pages": [
                {
                    "page_number": page.page_number,
                    "confidence": page.confidence,
                    "detected_handwriting": page.detected_handwriting,
                    "fields": page.fields,
                    "warnings": page.warnings,
                }
                for page in ocr_result.pages
            ],
        }
        investigation.ocr_status = "completed"
        investigation.embedding_status = "processing"
        db.commit()

        db.query(EvidenceChunk).filter(EvidenceChunk.investigation_id == investigation.id).delete()
        chunks = chunk_text(ocr_result.full_text)
        for index, content in enumerate(chunks):
            embedding = embedding_service.embed_text(content)
            db.add(
                EvidenceChunk(
                    investigation_id=investigation.id,
                    chunk_index=index,
                    content=content,
                    source_page=None,
                    confidence=investigation.ocr_confidence,
                    embedding=embedding,
                )
            )

        investigation.embedding_status = "completed"
        investigation.processing_status = "completed"
        db.commit()
    except Exception as exc:
        logger.exception("Processing failed for investigation %s", investigation_id)
        _set_failed(investigation, exc)
        db.commit()
