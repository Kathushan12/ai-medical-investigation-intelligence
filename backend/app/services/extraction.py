import json
import re
from typing import Any

from openai import OpenAI

from app.config import settings
from app.services.ocr import DocumentOcrResult


IMPORTANT_FIELDS = [
    "patient_name",
    "medical_condition",
    "incident_date",
    "hospital_location",
    "severity_level",
]


FIELD_ALIASES = {
    "patient_name": [
        "Patient Name",
        "Patient",
        "Pt Name",
        "Name",
    ],
    "medical_condition": [
        "Medical Condition",
        "Condition",
        "Diagnosis",
        "Suspected Medical Condition",
        "Main Complaint",
    ],
    "incident_date": [
        "Date",
        "Date of Investigation",
        "Investigation Date",
        "Report Date",
    ],
    "hospital_location": [
        "Hospital / Location",
        "Hospital",
        "Location",
        "Place",
        "Facility",
    ],
    "severity_level": [
        "Severity Level",
        "Severity",
        "Risk Level",
    ],
    "doctor_notes": [
        "Doctor Notes",
        "Doctor Note",
        "Clinical Notes",
        "Doctor Remarks",
    ],
    "lab_test_details": [
        "Lab/Test Details",
        "Lab Test Details",
        "Test Details",
        "Investigation Details",
        "Test Results",
        "Examination",
        "Radiology Findings",
    ],
    "summary": [
        "Summary",
        "Investigation Summary",
        "Report Summary",
    ],
}


STOP_LABELS = [
    "Patient Name",
    "Patient",
    "Patient ID",
    "Pt Name",
    "Name",
    "Date",
    "Date of Investigation",
    "Investigation Date",
    "Report Date",
    "Hospital / Location",
    "Hospital",
    "Location",
    "Place",
    "Condition",
    "Medical Condition",
    "Diagnosis",
    "Severity",
    "Severity Level",
    "Oxygen Sat",
    "Oxygen Sat.",
    "Examination",
    "Radiology Findings",
    "Lab/Test Details",
    "Lab Test Details",
    "Doctor Notes",
    "Doctor Note",
    "Summary",
    "Investigation Summary",
]


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _extract_json(text: str) -> dict[str, Any]:
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


def _first_non_empty(page_fields: list[dict[str, Any]], key: str) -> str:
    for fields in page_fields:
        value = fields.get(key)
        if value:
            return str(value).strip()
    return ""


def _normalize_ocr_spacing(text: str) -> str:
    """
    Fix OCR output where labels may be spaced like:
    P A T I E N T  N A M E
    """
    replacements = {
        r"P\s*A\s*T\s*I\s*E\s*N\s*T\s+N\s*A\s*M\s*E": "PATIENT NAME",
        r"P\s*A\s*T\s*I\s*E\s*N\s*T\s+I\s*D": "PATIENT ID",
        r"C\s*O\s*N\s*D\s*I\s*T\s*I\s*O\s*N": "CONDITION",
        r"H\s*O\s*S\s*P\s*I\s*T\s*A\s*L": "HOSPITAL",
        r"S\s*E\s*V\s*E\s*R\s*I\s*T\s*Y": "SEVERITY",
        r"D\s*A\s*T\s*E": "DATE",
        r"E\s*X\s*A\s*M\s*I\s*N\s*A\s*T\s*I\s*O\s*N": "EXAMINATION",
        r"O\s*X\s*Y\s*G\s*E\s*N\s+S\s*A\s*T": "OXYGEN SAT",
    }

    fixed = text

    for pattern, replacement in replacements.items():
        fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)

    return fixed


def _build_stop_pattern(exclude_labels: list[str]) -> str:
    stop_labels = [
        label
        for label in STOP_LABELS
        if _normalize_label(label) not in {_normalize_label(item) for item in exclude_labels}
    ]

    escaped = [re.escape(label) for label in stop_labels]
    return "|".join(escaped)


def _extract_between_labels(text: str, labels: list[str]) -> str:
    """
    Extract values from continuous OCR text.

    Example:
    PATIENT NAME Mohamed Rizwan PATIENT ID BT-RAD-5562
    returns Mohamed Rizwan
    """
    normalized_text = _normalize_ocr_spacing(text)
    normalized_text = re.sub(r"[ \t]+", " ", normalized_text)
    normalized_text = re.sub(r"\n+", "\n", normalized_text)

    for label in labels:
        stop_pattern = _build_stop_pattern([label])

        patterns = [
            rf"{re.escape(label)}\s*[:\-]\s*(.+?)(?=\s+(?:{stop_pattern})\b|$)",
            rf"{re.escape(label)}\s+(.+?)(?=\s+(?:{stop_pattern})\b|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, normalized_text, flags=re.IGNORECASE | re.DOTALL)

            if match:
                value = match.group(1).strip()
                value = re.sub(r"\s+", " ", value)

                if value:
                    return value

    return ""


def _value_from_line_or_next_line(text: str, labels: list[str]) -> str:
    fixed_text = _normalize_ocr_spacing(text)
    lines = [line.strip() for line in fixed_text.splitlines() if line.strip()]

    label_norms = {_normalize_label(label) for label in labels}
    stop_norms = {_normalize_label(label) for label in STOP_LABELS}

    for index, line in enumerate(lines):
        line_norm = _normalize_label(line)

        for label in labels:
            label_norm = _normalize_label(label)

            # Label exactly on one line, value on next line
            if line_norm == label_norm:
                for next_line in lines[index + 1 :]:
                    next_norm = _normalize_label(next_line)

                    if next_norm in stop_norms:
                        break

                    if next_line:
                        return next_line.strip()

            # Same line with colon
            same_line = re.search(
                rf"^\s*{re.escape(label)}\s*[:\-]\s*(.+)$",
                line,
                flags=re.IGNORECASE,
            )
            if same_line:
                return same_line.group(1).strip()

    return ""


def _value_from_label(text: str, labels: list[str]) -> str:
    return (
        _value_from_line_or_next_line(text, labels)
        or _extract_between_labels(text, labels)
    )


def _section_from_heading(text: str, headings: list[str]) -> str:
    fixed_text = _normalize_ocr_spacing(text)
    lines = [line.strip() for line in fixed_text.splitlines() if line.strip()]
    stop_norms = {_normalize_label(label) for label in STOP_LABELS}

    for index, line in enumerate(lines):
        line_norm = _normalize_label(line)

        for heading in headings:
            if line_norm == _normalize_label(heading):
                collected: list[str] = []

                for next_line in lines[index + 1 :]:
                    next_norm = _normalize_label(next_line)

                    if next_norm in stop_norms:
                        break

                    collected.append(next_line)

                return " ".join(collected).strip()

    return _extract_between_labels(text, headings)


def _normalize_severity(value: Any) -> str:
    severity = _safe_string(value)

    severity_map = {
        "low": "Low",
        "medium": "Medium",
        "moderate": "Medium",
        "high": "High",
        "critical": "Critical",
        "unknown": "Unknown",
    }

    return severity_map.get(severity.lower(), "Unknown")


def _normalize_fields(data: dict[str, Any], fallback_confidence: float) -> dict[str, Any]:
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
        "severity_level": _normalize_severity(data.get("severity_level")),
        "doctor_notes": _safe_string(data.get("doctor_notes")),
        "lab_test_details": _safe_string(data.get("lab_test_details")),
        "summary": _safe_string(data.get("summary")),
        "confidence": confidence,
        "warnings": warnings,
    }


def _merge_missing_fields(primary: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    merged = dict(primary)

    for key, fallback_value in fallback.items():
        if key in {"warnings", "confidence"}:
            continue

        if not merged.get(key) and fallback_value:
            merged[key] = fallback_value

    primary_warnings = primary.get("warnings", [])
    fallback_warnings = fallback.get("warnings", [])

    if not isinstance(primary_warnings, list):
        primary_warnings = [str(primary_warnings)]

    if not isinstance(fallback_warnings, list):
        fallback_warnings = [str(fallback_warnings)]

    merged["warnings"] = list(dict.fromkeys(primary_warnings + fallback_warnings))

    if not merged.get("confidence"):
        merged["confidence"] = fallback.get("confidence", 0.0)

    return merged


class StructuredExtractionService:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def extract(self, ocr_result: DocumentOcrResult) -> dict[str, Any]:
        page_fields = [page.fields for page in ocr_result.pages]
        text = ocr_result.full_text or ""

        fallback_fields = self._fallback_extract(
            text=text,
            page_fields=page_fields,
            ocr_result=ocr_result,
        )

        if self.client:
            try:
                ai_fields = self._extract_with_ai(text=text, ocr_result=ocr_result)
                merged_fields = _merge_missing_fields(ai_fields, fallback_fields)
                return self._add_missing_field_warnings(merged_fields)
            except Exception as exc:
                fallback_fields["warnings"].append(
                    f"AI structured extraction failed. Fallback extraction used. Error: {str(exc)}"
                )
                return self._add_missing_field_warnings(fallback_fields)

        return self._add_missing_field_warnings(fallback_fields)

    def _fallback_extract(
        self,
        text: str,
        page_fields: list[dict[str, Any]],
        ocr_result: DocumentOcrResult,
    ) -> dict[str, Any]:
        fixed_text = _normalize_ocr_spacing(text)

        patient_name = (
            _first_non_empty(page_fields, "patient_name")
            or _value_from_label(fixed_text, FIELD_ALIASES["patient_name"])
        )

        medical_condition = (
            _first_non_empty(page_fields, "medical_condition")
            or _value_from_label(fixed_text, FIELD_ALIASES["medical_condition"])
        )

        incident_date = (
            _first_non_empty(page_fields, "incident_date")
            or _value_from_label(fixed_text, FIELD_ALIASES["incident_date"])
        )

        hospital_location = (
            _first_non_empty(page_fields, "hospital_location")
            or _value_from_label(fixed_text, FIELD_ALIASES["hospital_location"])
        )

        severity_level = (
            _first_non_empty(page_fields, "severity_level")
            or _value_from_label(fixed_text, FIELD_ALIASES["severity_level"])
            or "Unknown"
        )

        doctor_notes = (
            _first_non_empty(page_fields, "doctor_notes")
            or _section_from_heading(fixed_text, FIELD_ALIASES["doctor_notes"])
        )

        lab_test_details = (
            _first_non_empty(page_fields, "lab_test_details")
            or _section_from_heading(fixed_text, FIELD_ALIASES["lab_test_details"])
        )

        summary = (
            _section_from_heading(fixed_text, FIELD_ALIASES["summary"])
            or fixed_text[:500]
        )

        return {
            "patient_name": patient_name,
            "medical_condition": medical_condition,
            "incident_date": incident_date,
            "hospital_location": hospital_location,
            "severity_level": _normalize_severity(severity_level),
            "doctor_notes": doctor_notes,
            "lab_test_details": lab_test_details,
            "summary": summary,
            "confidence": ocr_result.average_confidence,
            "warnings": list(ocr_result.metadata.get("warnings", [])),
        }

    def _extract_with_ai(self, text: str, ocr_result: DocumentOcrResult) -> dict[str, Any]:
        prompt = f"""
You are extracting structured fields from a medical investigation OCR result.

Use only the OCR text provided below.
Do not guess.

Important layout rule:
Some documents use labels and values in columns, such as:
PATIENT NAME
Mohamed Rizwan

or:
PATIENT NAME Mohamed Rizwan PATIENT ID BT-RAD-5562

Extract the value that follows the label until the next known label.

Map:
- CONDITION -> medical_condition
- HOSPITAL -> hospital_location
- DATE -> incident_date
- SEVERITY -> severity_level

Return ONLY valid JSON:
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

        return _normalize_fields(data, fallback_confidence=ocr_result.average_confidence)

    def _add_missing_field_warnings(self, fields: dict[str, Any]) -> dict[str, Any]:
        warnings = fields.get("warnings", [])

        if not isinstance(warnings, list):
            warnings = [str(warnings)]

        missing_fields = [
            field for field in IMPORTANT_FIELDS if not fields.get(field)
        ]

        if missing_fields:
            warnings.append(
                f"Important fields missing: {', '.join(missing_fields)}."
            )

        fields["warnings"] = list(dict.fromkeys(warnings))

        return fields


structured_extraction_service = StructuredExtractionService()