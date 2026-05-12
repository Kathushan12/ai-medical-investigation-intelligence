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


FORBIDDEN_EXAMPLE_VALUES = {
    "john doe",
    "jane doe",
    "john silva",
    "sample patient",
    "test patient",
    "example patient",
}


LAB_REPORT_KEYWORDS = [
    "department of biochemistry",
    "chemistry",
    "sample id",
    "sample type",
    "serum",
    "crea",
    "uric acid",
    "urea",
    "crp",
    "ref range",
    "collection date",
    "test date/time",
    "ordering date/time",
    "print date/time",
]


def _set_failed(investigation: Investigation, error: Exception) -> None:
    """
    Mark investigation as failed if OCR, extraction, chunking, or embedding fails.
    """
    if investigation.ocr_status == "processing":
        investigation.ocr_status = "failed"

    if investigation.embedding_status == "processing":
        investigation.embedding_status = "failed"

    investigation.processing_status = "failed"
    investigation.error_message = str(error)


def _safe_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value]

    return [str(value)]


def _average(values: list[Any]) -> float | None:
    clean_values: list[float] = []

    for value in values:
        if value is None:
            continue

        try:
            clean_values.append(float(value))
        except Exception:
            continue

    if not clean_values:
        return None

    return round(sum(clean_values) / len(clean_values), 3)


def _normalize_for_support_check(value: str) -> str:
    """
    Normalize text for simple support checking.
    Example:
    'John Doe' -> 'johndoe'
    """
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _is_forbidden_example_value(value: str | None) -> bool:
    if not value:
        return False

    return str(value).strip().lower() in FORBIDDEN_EXAMPLE_VALUES


def _is_supported_by_ocr_text(ocr_text: str, value: str | None) -> bool:
    """
    A structured field is accepted only if it is supported by OCR text.

    Example:
    If extracted patient_name = John Doe,
    but OCR text does not contain John Doe,
    then the field is cleared.
    """
    if not value:
        return True

    value = str(value).strip()

    if value.lower() in {"unknown", "not available", "n/a", "-"}:
        return True

    normalized_text = _normalize_for_support_check(ocr_text)
    normalized_value = _normalize_for_support_check(value)

    if not normalized_value:
        return True

    return normalized_value in normalized_text


def _looks_like_lab_report(ocr_text: str) -> bool:
    text = str(ocr_text or "").lower()
    return any(keyword in text for keyword in LAB_REPORT_KEYWORDS)


def _remove_unsupported_extracted_fields(
    fields: dict[str, Any],
    ocr_text: str,
) -> dict[str, Any]:
    """
    Prevent hallucinated fields from being stored.

    This protects against cases where the AI gives fake values like:
    John Doe, Acute Bronchitis, City General Hospital,
    even when those values are not in the uploaded document.
    """
    warnings = _safe_list(fields.get("warnings", []))
    unsupported_fields: list[str] = []

    for field_name in [
        "patient_name",
        "medical_condition",
        "incident_date",
        "hospital_location",
    ]:
        value = fields.get(field_name)

        if value and _is_forbidden_example_value(str(value)):
            unsupported_fields.append(field_name)
            fields[field_name] = ""
            warnings.append(
                f"Cleared forbidden example value from {field_name}: {value}"
            )
            continue

        if value and not _is_supported_by_ocr_text(ocr_text, str(value)):
            unsupported_fields.append(field_name)
            fields[field_name] = ""
            warnings.append(
                f"Cleared unsupported extracted field because it was not found in OCR text: {field_name}"
            )

    if _looks_like_lab_report(ocr_text):
        condition = str(fields.get("medical_condition") or "").strip()

        # For lab reports, do not infer diseases from lab values.
        # Only keep medical_condition if the condition is explicitly present in OCR text.
        if condition and not _is_supported_by_ocr_text(ocr_text, condition):
            fields["medical_condition"] = ""
            warnings.append(
                "Cleared inferred medical_condition because the uploaded document appears to be a lab report and no explicit diagnosis was found."
            )

        # A lab report usually has no severity unless explicitly written.
        severity = str(fields.get("severity_level") or "").strip()

        if severity.lower() not in {"low", "medium", "high", "critical"}:
            fields["severity_level"] = "Unknown"

    if unsupported_fields:
        warnings.append(
            "Unsupported extracted fields were cleared: "
            + ", ".join(sorted(set(unsupported_fields)))
        )

    fields["warnings"] = list(dict.fromkeys(warnings))
    return fields


def _build_retrieval_text(
    investigation: Investigation,
    ocr_text: str,
    fields: dict[str, Any],
) -> str:
    """
    Build a retrieval-friendly document for embeddings.

    This improves assistant answers because the vector chunks include
    both structured metadata and OCR text.
    """
    return f"""
Investigation File: {investigation.original_filename}

Structured Medical Fields:
Patient Name: {fields.get("patient_name") or ""}
Medical Condition: {fields.get("medical_condition") or ""}
Incident Date: {fields.get("incident_date") or ""}
Hospital / Location: {fields.get("hospital_location") or ""}
Severity Level: {fields.get("severity_level") or "Unknown"}
Doctor Notes: {fields.get("doctor_notes") or ""}
Lab/Test Details: {fields.get("lab_test_details") or ""}
Summary: {fields.get("summary") or ""}

OCR Extracted Text:
{ocr_text or ""}
""".strip()


def process_investigation_task(investigation_id: str) -> None:
    """
    Background task entry point.

    FastAPI calls this after upload.
    A new DB session is created because background tasks should not reuse
    the request session.
    """
    db = SessionLocal()

    try:
        process_investigation(db, uuid.UUID(investigation_id))
    finally:
        db.close()


def process_investigation(db: Session, investigation_id: uuid.UUID) -> None:
    """
    Complete AI processing workflow.

    Steps:
    1. Mark record as processing.
    2. Run AI OCR.
    3. Run structured extraction.
    4. Remove unsupported / hallucinated extracted fields.
    5. Compare OCR output with structured extraction.
    6. Decide whether human review is needed.
    7. Store OCR metadata, warnings, quality score, blur score, and fields.
    8. Chunk retrieval text.
    9. Generate embeddings.
    10. Store chunks in PostgreSQL + pgvector.
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
        # 4. Clear hallucinated / unsupported fields
        # ---------------------------------------------------------
        fields = _remove_unsupported_extracted_fields(
            fields=fields,
            ocr_text=ocr_result.full_text,
        )

        # ---------------------------------------------------------
        # 5. Compare OCR page fields with final structured fields
        # ---------------------------------------------------------
        validation_warnings = compare_ocr_with_structured_extraction(
            ocr_result=ocr_result,
            extracted_fields=fields,
        )

        # ---------------------------------------------------------
        # 6. Human review decision
        # ---------------------------------------------------------
        review_required, review_reason = build_human_review_decision(
            ocr_result=ocr_result,
            extracted_fields=fields,
            validation_warnings=validation_warnings,
        )

        # Force review if critical fields were cleared or missing
        extraction_warnings = _safe_list(fields.get("warnings", []))
        if any("Cleared unsupported" in warning for warning in extraction_warnings):
            review_required = True
            review_reason = (
                (review_reason or "")
                + " Unsupported or hallucinated fields were cleared before saving."
            ).strip()

        # ---------------------------------------------------------
        # 7. Build page metadata and quality information
        # ---------------------------------------------------------
        page_metadata: list[dict[str, Any]] = []

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
            extraction_warnings + _safe_list(validation_warnings)
        )

        # ---------------------------------------------------------
        # 8. Store final extracted data
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

        investigation.extraction_metadata = {
            "ocr": ocr_result.metadata,
            "structured_extraction_warnings": extraction_warnings,
            "validation_warnings": validation_warnings,
            "review_required": review_required,
            "review_reason": review_reason,
            "pages": page_metadata,
        }

        investigation.ocr_status = "completed"
        investigation.embedding_status = "processing"

        db.commit()

        # ---------------------------------------------------------
        # 9. Delete old chunks before creating new chunks
        # ---------------------------------------------------------
        db.query(EvidenceChunk).filter(
            EvidenceChunk.investigation_id == investigation.id
        ).delete()

        # ---------------------------------------------------------
        # 10. Build retrieval text, chunk it, and generate embeddings
        # ---------------------------------------------------------
        retrieval_text = _build_retrieval_text(
            investigation=investigation,
            ocr_text=ocr_result.full_text,
            fields=fields,
        )

        chunks = chunk_text(retrieval_text)

        if not chunks:
            chunks = [retrieval_text]

        for index, content in enumerate(chunks):
            if not content.strip():
                continue

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
        # 11. Complete processing
        # ---------------------------------------------------------
        investigation.embedding_status = "completed"
        investigation.processing_status = "completed"

        db.commit()

    except Exception as exc:
        logger.exception("Processing failed for investigation %s", investigation_id)

        _set_failed(investigation, exc)

        db.commit()