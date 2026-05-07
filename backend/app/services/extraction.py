import json
import re
from typing import Any

from openai import OpenAI

from app.config import settings
from app.services.ocr import DocumentOcrResult


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


def _extract_json(text: str) -> dict[str, Any]:
    """
    Safely extract JSON even if the model returns markdown code blocks.
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"^```", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def _chat_response_text(response: Any) -> str:
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _normalize_fields(data: dict[str, Any], fallback_confidence: float) -> dict[str, Any]:
    severity = _safe_string(data.get("severity_level")) or "Unknown"

    if severity not in {"Low", "Medium", "High", "Critical", "Unknown"}:
        severity = "Unknown"

    confidence = data.get("confidence", fallback_confidence)

    try:
        confidence = float(confidence)
    except Exception:
        confidence = fallback_confidence

    confidence = max(0.0, min(1.0, confidence))

    warnings = data.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = [str(warnings)]

    return {
        "patient_name": _safe_string(data.get("patient_name")),
        "medical_condition": _safe_string(data.get("medical_condition")),
        "incident_date": _safe_string(data.get("incident_date")),
        "hospital_location": _safe_string(data.get("hospital_location")),
        "severity_level": severity,
        "doctor_notes": _safe_string(data.get("doctor_notes")),
        "lab_test_details": _safe_string(data.get("lab_test_details")),
        "summary": _safe_string(data.get("summary")),
        "confidence": confidence,
        "warnings": warnings,
    }


class StructuredExtractionService:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def extract(self, ocr_result: DocumentOcrResult) -> dict[str, Any]:
        page_fields = [page.fields for page in ocr_result.pages]
        text = ocr_result.full_text or ""

        if self.client:
            return self._extract_with_ai(text=text, ocr_result=ocr_result)

        return self._fallback_extract(text=text, page_fields=page_fields, ocr_result=ocr_result)

    def _fallback_extract(
        self,
        text: str,
        page_fields: list[dict[str, Any]],
        ocr_result: DocumentOcrResult,
    ) -> dict[str, Any]:
        return {
            "patient_name": _first_non_empty(page_fields, "patient_name")
            or _regex_value(text, "Patient Name"),
            "medical_condition": _first_non_empty(page_fields, "medical_condition")
            or _regex_value(text, "Medical Condition"),
            "incident_date": _first_non_empty(page_fields, "incident_date")
            or _regex_value(text, "Date"),
            "hospital_location": _first_non_empty(page_fields, "hospital_location")
            or _regex_value(text, "Hospital / Location")
            or _regex_value(text, "Hospital"),
            "severity_level": _first_non_empty(page_fields, "severity_level")
            or _regex_value(text, "Severity Level")
            or "Unknown",
            "doctor_notes": _first_non_empty(page_fields, "doctor_notes")
            or _regex_value(text, "Doctor Notes"),
            "lab_test_details": _first_non_empty(page_fields, "lab_test_details")
            or _regex_value(text, "Lab/Test Details")
            or _regex_value(text, "Lab Test Details"),
            "summary": _regex_value(text, "Summary") or text[:500],
            "confidence": ocr_result.average_confidence,
            "warnings": ocr_result.metadata.get("warnings", []),
        }

    def _extract_with_ai(self, text: str, ocr_result: DocumentOcrResult) -> dict[str, Any]:
        prompt = f"""
You are extracting structured fields from a medical investigation OCR result.

Use only the OCR text provided below.
Do not guess.
Use empty strings for missing fields.

Return ONLY valid JSON with this exact structure:
{{
  "patient_name": "",
  "medical_condition": "",
  "incident_date": "",
  "hospital_location": "",
  "severity_level": "Unknown",
  "doctor_notes": "",
  "lab_test_details": "",
  "summary": "",
  "confidence": 0.0,
  "warnings": []
}}

Rules:
- severity_level must be one of: Low, Medium, High, Critical, Unknown.
- confidence must be between 0 and 1.
- summary must be short and based only on the OCR text.
- warnings must include uncertain or missing important fields.

OCR confidence: {ocr_result.average_confidence}

OCR text:
{text[:15000]}
""".strip()

        response = self.client.chat.completions.create(
            model=settings.openai_text_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful medical investigation structured data extraction assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw = _chat_response_text(response)
        data = _extract_json(raw)

        return _normalize_fields(
            data=data,
            fallback_confidence=ocr_result.average_confidence,
        )


structured_extraction_service = StructuredExtractionService()