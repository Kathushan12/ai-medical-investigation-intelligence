import json
import re
from typing import Any

from openai import OpenAI

from app.config import settings
from app.services.ocr import DocumentOcrResult


EXTRACTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "patient_name": {"type": "string"},
        "medical_condition": {"type": "string"},
        "incident_date": {"type": "string"},
        "hospital_location": {"type": "string"},
        "severity_level": {"type": "string", "enum": ["Low", "Medium", "High", "Critical", "Unknown"]},
        "doctor_notes": {"type": "string"},
        "lab_test_details": {"type": "string"},
        "summary": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "patient_name",
        "medical_condition",
        "incident_date",
        "hospital_location",
        "severity_level",
        "doctor_notes",
        "lab_test_details",
        "summary",
        "confidence",
        "warnings",
    ],
}


def _response_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
    return "\n".join(parts)


def _first_non_empty(page_fields: list[dict[str, Any]], key: str) -> str:
    for fields in page_fields:
        value = fields.get(key)
        if value:
            return str(value)
    return ""


def _regex_value(text: str, label: str) -> str:
    pattern = rf"{re.escape(label)}\s*[:\-]\s*(.+)"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


class StructuredExtractionService:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def extract(self, ocr_result: DocumentOcrResult) -> dict[str, Any]:
        page_fields = [page.fields for page in ocr_result.pages]
        text = ocr_result.full_text

        if self.client:
            return self._extract_with_ai(text=text, ocr_result=ocr_result)

        # Development fallback for TXT samples. Real evaluation should use the AI path.
        return {
            "patient_name": _first_non_empty(page_fields, "patient_name") or _regex_value(text, "Patient Name"),
            "medical_condition": _first_non_empty(page_fields, "medical_condition") or _regex_value(text, "Medical Condition"),
            "incident_date": _first_non_empty(page_fields, "incident_date") or _regex_value(text, "Date"),
            "hospital_location": _first_non_empty(page_fields, "hospital_location") or _regex_value(text, "Hospital / Location"),
            "severity_level": _first_non_empty(page_fields, "severity_level") or _regex_value(text, "Severity Level") or "Unknown",
            "doctor_notes": _first_non_empty(page_fields, "doctor_notes") or _regex_value(text, "Doctor Notes"),
            "lab_test_details": _first_non_empty(page_fields, "lab_test_details") or _regex_value(text, "Lab/Test Details"),
            "summary": _regex_value(text, "Summary") or text[:500],
            "confidence": ocr_result.average_confidence,
            "warnings": ocr_result.metadata.get("warnings", []),
        }

    def _extract_with_ai(self, text: str, ocr_result: DocumentOcrResult) -> dict[str, Any]:
        prompt = f"""
You are extracting structured fields from a medical investigation OCR result.
Use only the text provided. Do not guess. Use empty strings for missing fields.
Return a short summary and confidence score.

OCR confidence: {ocr_result.average_confidence}
OCR text:
{text[:15000]}
""".strip()
        response = self.client.responses.create(
            model=settings.openai_text_model,
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "medical_investigation_fields",
                    "strict": True,
                    "schema": EXTRACTION_SCHEMA,
                }
            },
        )
        return json.loads(_response_text(response))


structured_extraction_service = StructuredExtractionService()
