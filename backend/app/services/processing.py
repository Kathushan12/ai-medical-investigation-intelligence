import logging
import uuid
from typing import Any

from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import EvidenceChunk, Investigation
from app.services.chunking import chunk_text
from app.services.embeddings import embedding_service
from app.services.extraction import structured_extraction_service
from app.services.ocr import ai_ocr_service
from app.services.ocr_validation import (
    build_human_review_decision,
    compare_ocr_with_structured_extraction,
)

logger = logging.getLogger(__name__)


def _set_failed(investigation: Investigation, error: Exception) -> None:
    """
    Mark the investigation as failed if any step in the AI pipeline fails.
    """
    if investigation.ocr_status == "processing":
        investigation.ocr_status = "failed"

    if investigation.embedding_status == "processing":
        investigation.embedding_status = "failed"

    investigation.processing_status = "failed"
    investigation.error_message = str(error)


def _safe_list(value: Any) -> list[str]:
    """
    Ensure warning fields are always stored as a list of strings.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    return [str(value)]


def _average(values: list[float]) -> float | None:
    """
    Calculate average safely.
    """
    clean_values = [float(value) for value in values if value is not None]

    if not clean_values:
        return None

    return round(sum(clean_values) / len(clean_values), 3)


def process_investigation_task(investigation_id: str) -> None:
    """
    Background task entry point.

    FastAPI calls this after file upload. It opens a new DB session because
    background tasks should not reuse the request session.
    """
    db = SessionLocal()

    try:
        process_investigation(db, uuid.UUID(investigation_id))
    finally:
        db.close()


def process_investigation(db: Session, investigation_id: uuid.UUID) -> None:
    """
    Full AI processing pipeline.

    Steps:
    1. Mark investigation as processing.
    2. Run AI OCR.
    3. Run structured extraction.
    4. Validate OCR output against extracted fields.
    5. Store OCR warnings, metadata, confidence, quality score, and review decision.
    6. Chunk extracted text.
    7. Generate embeddings.
    8. Store chunks and embeddings.
    9. Mark investigation as completed.
    """
    investigation = db.get(Investigation, investigation_id)

    if not investigation:
        raise ValueError("Investigation not found")

    try:
        # ---------------------------------------------------------
        # 1. Start processing
        # ---------------------------------------------------------
        investigation.processing_status = "processing"
        investigation.ocr_status = "processing"
        investigation.embedding_status = "pending"
        investigation.error_message = None

        # Reset review and warning fields when reprocessing
        investigation.ocr_warnings = []
        investigation.ocr_metadata = {}
        investigation.extraction_warnings = []
        investigation.review_required = False
        investigation.review_reason = None
        investigation.document_quality_score = None
        investigation.blur_score = None

        db.commit()

        # ---------------------------------------------------------
        # 2. AI OCR
        # ---------------------------------------------------------
        ocr_result = ai_ocr_service.process_document(investigation.file_path)

        # ---------------------------------------------------------
        # 3. Structured extraction
        # ---------------------------------------------------------
        fields = structured_extraction_service.extract(ocr_result)

        # ---------------------------------------------------------
        # 4. OCR vs structured extraction validation
        # ---------------------------------------------------------
        validation_warnings = compare_ocr_with_structured_extraction(
            ocr_result=ocr_result,
            extracted_fields=fields,
        )

        review_required, review_reason = build_human_review_decision(
            ocr_result=ocr_result,
            extracted_fields=fields,
            validation_warnings=validation_warnings,
        )

        # ---------------------------------------------------------
        # 5. Build OCR quality metadata
        # ---------------------------------------------------------
        page_metadata = []

        for page in ocr_result.pages:
            page_metadata.append(
                {
                    "page_number": page.page_number,
                    "confidence": page.confidence,
                    "detected_handwriting": page.detected_handwriting,
                    "fields": page.fields,
                    "warnings": page.warnings,
                    "metadata": page.metadata,
                }
            )

        blur_scores = [
            page.metadata.get("final_blur_score")
            for page in ocr_result.pages
            if page.metadata.get("final_blur_score") is not None
        ]

        average_blur_score = _average(blur_scores)

        all_ocr_warnings = _safe_list(ocr_result.metadata.get("warnings", []))
        all_extraction_warnings = (
            _safe_list(fields.get("warnings", []))
            + _safe_list(validation_warnings)
        )

        # ---------------------------------------------------------
        # 6. Store extracted investigation data
        # ---------------------------------------------------------
        investigation.extracted_text = ocr_result.full_text
        investigation.ocr_confidence = (
            fields.get("confidence") or ocr_result.average_confidence
        )

        investigation.patient_name = fields.get("patient_name") or None
        investigation.medical_condition = fields.get("medical_condition") or None
        investigation.incident_date = fields.get("incident_date") or None
        investigation.hospital_location = fields.get("hospital_location") or None
        investigation.severity_level = fields.get("severity_level") or "Unknown"
        investigation.doctor_notes = fields.get("doctor_notes") or None
        investigation.lab_test_details = fields.get("lab_test_details") or None
        investigation.summary = fields.get("summary") or None

        # ---------------------------------------------------------
        # 7. Store OCR improvement fields
        # ---------------------------------------------------------
        investigation.ocr_warnings = all_ocr_warnings
        investigation.ocr_metadata = {
            **ocr_result.metadata,
            "pages": page_metadata,
        }
        investigation.extraction_warnings = all_extraction_warnings

        investigation.document_quality_score = ocr_result.metadata.get(
            "average_quality_score"
        )
        investigation.blur_score = average_blur_score

        investigation.review_required = review_required
        investigation.review_reason = review_reason

        # Keep old metadata field also updated for compatibility
        investigation.extraction_metadata = {
            "ocr": ocr_result.metadata,
            "structured_extraction_warnings": fields.get("warnings", []),
            "validation_warnings": validation_warnings,
            "review_required": review_required,
            "review_reason": review_reason,
            "pages": page_metadata,
        }

        investigation.ocr_status = "completed"
        investigation.embedding_status = "processing"

        db.commit()

        # ---------------------------------------------------------
        # 8. Delete old chunks before creating new chunks
        # ---------------------------------------------------------
        db.query(EvidenceChunk).filter(
            EvidenceChunk.investigation_id == investigation.id
        ).delete()

        # ---------------------------------------------------------
        # 9. Chunk text and generate embeddings
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # 10. Complete processing
        # ---------------------------------------------------------
        investigation.embedding_status = "completed"
        investigation.processing_status = "completed"

        db.commit()

    except Exception as exc:
        logger.exception("Processing failed for investigation %s", investigation_id)

        _set_failed(investigation, exc)

        db.commit()