from typing import Any

from app.config import settings
from app.services.ocr import DocumentOcrResult


IMPORTANT_FIELDS = [
    "patient_name",
    "medical_condition",
    "incident_date",
    "hospital_location",
    "severity_level",
]


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def compare_ocr_with_structured_extraction(
    ocr_result: DocumentOcrResult,
    extracted_fields: dict[str, Any],
) -> list[str]:
    """
    Compare page-level OCR fields with final structured extraction.
    This helps detect possible AI inconsistencies.
    """
    warnings: list[str] = []

    page_fields: list[dict[str, Any]] = [
        page.fields for page in ocr_result.pages if page.fields
    ]

    for field in IMPORTANT_FIELDS:
        extracted_value = _normalize(extracted_fields.get(field))

        if not extracted_value:
            warnings.append(f"Structured extraction missing important field: {field}.")
            continue

        ocr_values = {
            _normalize(page_field.get(field))
            for page_field in page_fields
            if _normalize(page_field.get(field))
        }

        if ocr_values and extracted_value not in ocr_values:
            warnings.append(
                f"Mismatch detected for {field}. OCR values={list(ocr_values)}, extracted value={extracted_value}."
            )

    return warnings


def build_human_review_decision(
    ocr_result: DocumentOcrResult,
    extracted_fields: dict[str, Any],
    validation_warnings: list[str],
) -> tuple[bool, str]:
    reasons: list[str] = []

    if ocr_result.average_confidence < settings.min_ocr_confidence:
        reasons.append(
            f"OCR confidence is below threshold. Confidence={ocr_result.average_confidence:.2f}, threshold={settings.min_ocr_confidence:.2f}."
        )

    if ocr_result.metadata.get("review_required"):
        reasons.append("OCR preprocessing or page-level extraction recommended human review.")

    if validation_warnings:
        reasons.append("OCR output and structured extraction have possible inconsistencies.")

    missing_required = []

    for field in IMPORTANT_FIELDS:
        if not extracted_fields.get(field):
            missing_required.append(field)

    if missing_required:
        reasons.append(f"Important fields missing: {', '.join(missing_required)}.")

    review_required = len(reasons) > 0
    review_reason = " ".join(reasons)

    return review_required, review_reason