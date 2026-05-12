import json
import re
from typing import Any

from openai import OpenAI

from app.config import settings
from app.services.ocr import DocumentOcrResult


IMPORTANT_FIELDS = [
    "patient_name",
    "incident_date",
    "hospital_location",
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
        "Collection Date",
        "Ordering Date/Time",
        "Test Date/Time",
        "Print Date/Time",
    ],
    "hospital_location": [
        "Hospital / Location",
        "Hospital",
        "Location",
        "Place",
        "Facility",
        "Lab",
        "Laboratory",
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
        "Medical Notes",
        "Physician",
        "Comment",
    ],
    "lab_test_details": [
        "Lab/Test Details",
        "Lab Test Details",
        "Test Details",
        "Investigation Details",
        "Test Results",
        "Test Type",
        "Chemistry",
        "Result",
        "Unit",
        "Ref Range",
        "Examination",
        "Radiology Findings",
        "Peak Flow",
        "Oxygen Sat",
        "Resp. Rate",
        "Blood Pressure",
        "Troponin I",
        "A/G",
        "CREA",
        "Uric Acid",
        "UREA",
        "CRP",
        "Sample ID",
        "Sample Type",
        "Department",
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
    "Age",
    "Gender",
    "Date",
    "Date of Investigation",
    "Investigation Date",
    "Report Date",
    "Collection Date",
    "Collection Time",
    "Ordering Date/Time",
    "Test Date/Time",
    "Print Date/Time",
    "Hospital / Location",
    "Hospital",
    "Location",
    "Place",
    "Facility",
    "Lab",
    "Laboratory",
    "Condition",
    "Medical Condition",
    "Diagnosis",
    "Severity",
    "Severity Level",
    "Risk Level",
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
    "Clinical Findings",
    "Treatment Advice",
    "Recommended Action",
    "Sample ID",
    "Sample Type",
    "Department",
    "Comment",
    "Chemistry",
    "Result",
    "Flag",
    "Ref Range",
    "MLT",
    "Review Officer",
]


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
    "biochemistry",
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


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _normalize_support_text(value: str) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def _is_forbidden_example_value(value: str | None) -> bool:
    if not value:
        return False

    return str(value).strip().lower() in FORBIDDEN_EXAMPLE_VALUES


def _is_supported_by_text(text: str, value: str | None) -> bool:
    if not value:
        return True

    value = str(value).strip()

    if value.lower() in {"unknown", "not available", "n/a", "-"}:
        return True

    normalized_text = _normalize_support_text(text)
    normalized_value = _normalize_support_text(value)

    if not normalized_value:
        return True

    return normalized_value in normalized_text


def _looks_like_lab_report(text: str) -> bool:
    lower_text = str(text or "").lower()
    return any(keyword in lower_text for keyword in LAB_REPORT_KEYWORDS)


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


def _normalize_ocr_spacing(text: str) -> str:
    replacements = {
        r"P\s*A\s*T\s*I\s*E\s*N\s*T\s+N\s*A\s*M\s*E": "PATIENT NAME",
        r"P\s*A\s*T\s*I\s*E\s*N\s*T\s+I\s*D": "PATIENT ID",
        r"C\s*O\s*N\s*D\s*I\s*T\s*I\s*O\s*N": "CONDITION",
        r"H\s*O\s*S\s*P\s*I\s*T\s*A\s*L": "HOSPITAL",
        r"P\s*L\s*A\s*C\s*E": "PLACE",
        r"S\s*E\s*V\s*E\s*R\s*I\s*T\s*Y": "SEVERITY",
        r"D\s*A\s*T\s*E": "DATE",
        r"C\s*O\s*L\s*L\s*E\s*C\s*T\s*I\s*O\s*N\s+D\s*A\s*T\s*E": "COLLECTION DATE",
        r"E\s*X\s*A\s*M\s*I\s*N\s*A\s*T\s*I\s*O\s*N": "EXAMINATION",
        r"O\s*X\s*Y\s*G\s*E\s*N\s+S\s*A\s*T": "OXYGEN SAT",
        r"P\s*E\s*A\s*K\s+F\s*L\s*O\s*W": "PEAK FLOW",
        r"R\s*E\s*S\s*P\s*\.?\s*R\s*A\s*T\s*E": "RESP. RATE",
    }

    fixed = text

    for pattern, replacement in replacements.items():
        fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)

    return fixed


def _build_stop_pattern(exclude_labels: list[str]) -> str:
    excluded = {_normalize_label(item) for item in exclude_labels}

    stop_labels = [
        label for label in STOP_LABELS if _normalize_label(label) not in excluded
    ]

    return "|".join(re.escape(label) for label in stop_labels)


def _value_between_labels(text: str, labels: list[str]) -> str:
    fixed = _normalize_ocr_spacing(text)
    fixed = re.sub(r"[ \t]+", " ", fixed)
    fixed = re.sub(r"\n+", "\n", fixed)

    for label in labels:
        stop_pattern = _build_stop_pattern([label])

        patterns = [
            rf"{re.escape(label)}\s*[:\-]\s*(.+?)(?=\s+(?:{stop_pattern})\b|$)",
            rf"{re.escape(label)}\s+(.+?)(?=\s+(?:{stop_pattern})\b|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, fixed, flags=re.IGNORECASE | re.DOTALL)

            if match:
                value = re.sub(r"\s+", " ", match.group(1)).strip()
                if value:
                    return value

    return ""


def _value_from_line_or_next_line(text: str, labels: list[str]) -> str:
    fixed = _normalize_ocr_spacing(text)
    lines = [line.strip() for line in fixed.splitlines() if line.strip()]

    stop_norms = {_normalize_label(label) for label in STOP_LABELS}

    for index, line in enumerate(lines):
        line_norm = _normalize_label(line)

        for label in labels:
            label_norm = _normalize_label(label)

            if line_norm == label_norm:
                for next_line in lines[index + 1 :]:
                    next_norm = _normalize_label(next_line)

                    if next_norm in stop_norms:
                        break

                    if next_line:
                        return next_line.strip()

            same_line = re.search(
                rf"^\s*{re.escape(label)}\s*[:\-]\s*(.+)$",
                line,
                flags=re.IGNORECASE,
            )

            if same_line:
                return same_line.group(1).strip()

    return ""


def _value_from_label(text: str, labels: list[str]) -> str:
    return _value_from_line_or_next_line(text, labels) or _value_between_labels(
        text, labels
    )


def _section_from_heading(text: str, headings: list[str]) -> str:
    fixed = _normalize_ocr_spacing(text)
    lines = [line.strip() for line in fixed.splitlines() if line.strip()]
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

    return _value_between_labels(text, headings)


def _first_supported_page_field(
    page_fields: list[dict[str, Any]],
    key: str,
    text: str,
) -> str:
    for fields in page_fields:
        value = _safe_string(fields.get(key))

        if not value:
            continue

        if _is_forbidden_example_value(value):
            continue

        if _is_supported_by_text(text, value):
            return value

    return ""


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


def _extract_lab_location(text: str) -> str:
    """
    Handles lab headers like:
    PATH LAB, DGH - HORANA
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:8]:
        lower_line = line.lower()

        if "lab" in lower_line or "hospital" in lower_line or "dgh" in lower_line:
            return line

    return ""


def _extract_lab_results(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    keywords = [
        "A/G",
        "CREA",
        "Uric Acid",
        "UREA",
        "CRP",
        "Sample ID",
        "Sample Type",
        "Collection Date",
        "Collection Time",
        "Department",
        "Comment",
        "Ordering Date/Time",
        "Test Date/Time",
        "Print Date/Time",
        "Chemistry",
        "Result",
        "Ref Range",
    ]

    collected: list[str] = []

    for line in lines:
        if any(keyword.lower() in line.lower() for keyword in keywords):
            collected.append(line)

    return " | ".join(collected).strip()


def _build_summary(fields: dict[str, Any], text: str) -> str:
    patient = fields.get("patient_name") or "Unknown patient"
    condition = fields.get("medical_condition") or "No explicit diagnosis found"
    location = fields.get("hospital_location") or "Unknown location"
    date = fields.get("incident_date") or "Unknown date"

    if _looks_like_lab_report(text):
        return (
            f"Laboratory report for {patient}. "
            f"Date: {date}. "
            f"Location/Lab: {location}. "
            f"Condition: {condition}."
        )

    return (
        f"Medical investigation report for {patient}. "
        f"Condition: {condition}. "
        f"Date: {date}. "
        f"Location: {location}."
    )


def _normalize_fields(
    data: dict[str, Any],
    fallback_confidence: float,
    ocr_text: str,
) -> dict[str, Any]:
    confidence = data.get("confidence", fallback_confidence)

    try:
        confidence = float(confidence)
    except Exception:
        confidence = fallback_confidence

    confidence = max(0.0, min(1.0, confidence))

    warnings = data.get("warnings", [])

    if not isinstance(warnings, list):
        warnings = [str(warnings)]

    fields = {
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

    fields = _clean_unsupported_fields(fields, ocr_text)

    if not fields.get("summary"):
        fields["summary"] = _build_summary(fields, ocr_text)

    return fields


def _clean_unsupported_fields(fields: dict[str, Any], ocr_text: str) -> dict[str, Any]:
    warnings = fields.get("warnings", [])

    if not isinstance(warnings, list):
        warnings = [str(warnings)]

    for field_name in [
        "patient_name",
        "medical_condition",
        "incident_date",
        "hospital_location",
    ]:
        value = fields.get(field_name)

        if value and _is_forbidden_example_value(str(value)):
            fields[field_name] = ""
            warnings.append(
                f"Cleared forbidden example value from {field_name}: {value}"
            )
            continue

        if value and not _is_supported_by_text(ocr_text, str(value)):
            fields[field_name] = ""
            warnings.append(
                f"Cleared unsupported extracted field because it was not found in OCR text: {field_name}"
            )

    if _looks_like_lab_report(ocr_text):
        condition = _safe_string(fields.get("medical_condition"))

        if condition and not _is_supported_by_text(ocr_text, condition):
            fields["medical_condition"] = ""
            warnings.append(
                "Cleared inferred medical_condition because the document appears to be a lab report and no diagnosis was explicitly visible."
            )

        if fields.get("severity_level") not in {"Low", "Medium", "High", "Critical"}:
            fields["severity_level"] = "Unknown"

    fields["warnings"] = list(dict.fromkeys(warnings))
    return fields


def _merge_missing_fields(
    primary: dict[str, Any],
    fallback: dict[str, Any],
    ocr_text: str,
) -> dict[str, Any]:
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

    merged = _clean_unsupported_fields(merged, ocr_text)

    if not merged.get("summary"):
        merged["summary"] = _build_summary(merged, ocr_text)

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
                ai_fields = self._extract_with_ai(
                    text=text,
                    ocr_result=ocr_result,
                )

                merged_fields = _merge_missing_fields(
                    primary=ai_fields,
                    fallback=fallback_fields,
                    ocr_text=text,
                )

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

        is_lab_report = _looks_like_lab_report(fixed_text)

        patient_name = (
            _first_supported_page_field(page_fields, "patient_name", fixed_text)
            or _value_from_label(fixed_text, FIELD_ALIASES["patient_name"])
        )

        incident_date = (
            _first_supported_page_field(page_fields, "incident_date", fixed_text)
            or _value_from_label(fixed_text, FIELD_ALIASES["incident_date"])
        )

        hospital_location = (
            _first_supported_page_field(page_fields, "hospital_location", fixed_text)
            or _value_from_label(fixed_text, FIELD_ALIASES["hospital_location"])
            or _extract_lab_location(fixed_text)
        )

        if is_lab_report:
            medical_condition = ""
            severity_level = "Unknown"
        else:
            medical_condition = (
                _first_supported_page_field(page_fields, "medical_condition", fixed_text)
                or _value_from_label(fixed_text, FIELD_ALIASES["medical_condition"])
            )

            severity_level = (
                _first_supported_page_field(page_fields, "severity_level", fixed_text)
                or _value_from_label(fixed_text, FIELD_ALIASES["severity_level"])
                or "Unknown"
            )

        doctor_notes = (
            _first_supported_page_field(page_fields, "doctor_notes", fixed_text)
            or _section_from_heading(fixed_text, FIELD_ALIASES["doctor_notes"])
        )

        lab_test_details = (
            _first_supported_page_field(page_fields, "lab_test_details", fixed_text)
            or _section_from_heading(fixed_text, FIELD_ALIASES["lab_test_details"])
            or _extract_lab_results(fixed_text)
        )

        fields = {
            "patient_name": patient_name,
            "medical_condition": medical_condition,
            "incident_date": incident_date,
            "hospital_location": hospital_location,
            "severity_level": _normalize_severity(severity_level),
            "doctor_notes": doctor_notes,
            "lab_test_details": lab_test_details,
            "summary": "",
            "confidence": ocr_result.average_confidence,
            "warnings": list(ocr_result.metadata.get("warnings", [])),
        }

        fields = _clean_unsupported_fields(fields, fixed_text)
        fields["summary"] = _build_summary(fields, fixed_text)

        return fields

    def _extract_with_ai(self, text: str, ocr_result: DocumentOcrResult) -> dict[str, Any]:
        prompt = f"""
You are extracting structured fields from an OCR result of a medical investigation document.

Use ONLY the OCR text provided below.
Do not use outside knowledge.
Do not guess.
Do not invent patient names, hospitals, dates, diagnoses, doctor notes, or lab results.
Never use example names such as John Doe, Jane Doe, John Silva, or sample patient names.

Important lab report rule:
If the OCR text appears to be a laboratory report and no diagnosis is explicitly written, set medical_condition to an empty string.
Do not infer diseases from values such as CRP, UREA, CREA, Uric Acid, glucose, or other lab results.
If severity is not explicitly written, set severity_level to "Unknown".

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

Field rules:
- patient_name must appear in OCR text.
- medical_condition must appear explicitly in OCR text.
- incident_date must appear in OCR text.
- hospital_location must appear in OCR text.
- lab_test_details must be based only on visible OCR text.
- summary must be based only on visible OCR text.

OCR confidence: {ocr_result.average_confidence}

OCR text:
{text[:15000]}
""".strip()

        response = self.client.chat.completions.create(
            model=settings.openai_text_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict medical document extraction assistant. You only extract values supported by OCR text.",
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
            ocr_text=text,
        )

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

        if not fields.get("medical_condition"):
            warnings.append(
                "Medical condition not found explicitly in OCR text."
            )

        fields["warnings"] = list(dict.fromkeys(warnings))

        return fields


structured_extraction_service = StructuredExtractionService()