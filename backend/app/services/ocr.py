import base64
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from openai import OpenAI
from PIL import Image

from app.config import settings
from app.services.image_preprocessing import preprocess_for_ocr


@dataclass
class PageOcrResult:
    page_number: int
    page_text: str
    confidence: float
    detected_handwriting: bool = False
    fields: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentOcrResult:
    full_text: str
    average_confidence: float
    pages: list[PageOcrResult]
    metadata: dict[str, Any]


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
        "Peak Flow",
        "Oxygen Sat",
        "Resp. Rate",
        "Blood Pressure",
        "Troponin I",
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
    "Facility",
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
    "Treatment Advice",
    "Recommended Action",
    "Unit",
    "Blood Pressure",
    "Troponin I",
    "Peak Flow",
    "Resp. Rate",
]


def _image_to_data_url(image: Image.Image, image_format: str = "PNG") -> str:
    image = image.convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def _render_pdf_pages(path: str) -> list[Image.Image]:
    pdf = pdfium.PdfDocument(path)
    images: list[Image.Image] = []
    max_pages = min(len(pdf), settings.max_pdf_pages)

    for page_index in range(max_pages):
        page = pdf[page_index]
        bitmap = page.render(scale=2.0).to_pil()
        images.append(bitmap.convert("RGB"))

    return images


def _load_images(path: str) -> list[Image.Image]:
    suffix = Path(path).suffix.lower()

    if suffix == ".pdf":
        return _render_pdf_pages(path)

    if suffix in {".png", ".jpg", ".jpeg"}:
        return [Image.open(path).convert("RGB")]

    return []


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


def _safe_severity(value: Any) -> str:
    severity = str(value or "Unknown").strip()

    severity_map = {
        "low": "Low",
        "medium": "Medium",
        "moderate": "Medium",
        "high": "High",
        "critical": "Critical",
        "unknown": "Unknown",
    }

    return severity_map.get(severity.lower(), "Unknown")


def _normalize_ocr_data(data: dict[str, Any]) -> dict[str, Any]:
    warnings = data.get("warnings", [])

    if not isinstance(warnings, list):
        warnings = [str(warnings)]

    try:
        confidence = float(data.get("confidence", 0.0) or 0.0)
    except Exception:
        confidence = 0.0

    confidence = max(0.0, min(1.0, confidence))

    return {
        "page_text": str(data.get("page_text", "")),
        "confidence": confidence,
        "detected_handwriting": bool(data.get("detected_handwriting", False)),
        "patient_name": str(data.get("patient_name", "")).strip(),
        "medical_condition": str(data.get("medical_condition", "")).strip(),
        "incident_date": str(data.get("incident_date", "")).strip(),
        "hospital_location": str(data.get("hospital_location", "")).strip(),
        "severity_level": _safe_severity(data.get("severity_level")),
        "doctor_notes": str(data.get("doctor_notes", "")).strip(),
        "lab_test_details": str(data.get("lab_test_details", "")).strip(),
        "evidence_type": str(data.get("evidence_type", "")).strip(),
        "warnings": warnings,
    }


def _combine_confidence(ai_confidence: float, image_quality_score: float) -> float:
    combined = (ai_confidence * 0.75) + (image_quality_score * 0.25)
    return round(max(0.0, min(1.0, combined)), 3)


def _normalize_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _normalize_ocr_spacing(text: str) -> str:
    """
    Fix OCR output where labels may be spaced or stylized.
    Example: P A T I E N T  N A M E -> PATIENT NAME
    """
    replacements = {
        r"P\s*A\s*T\s*I\s*E\s*N\s*T\s+N\s*A\s*M\s*E": "PATIENT NAME",
        r"P\s*A\s*T\s*I\s*E\s*N\s*T\s+I\s*D": "PATIENT ID",
        r"C\s*O\s*N\s*D\s*I\s*T\s*I\s*O\s*N": "CONDITION",
        r"H\s*O\s*S\s*P\s*I\s*T\s*A\s*L": "HOSPITAL",
        r"P\s*L\s*A\s*C\s*E": "PLACE",
        r"S\s*E\s*V\s*E\s*R\s*I\s*T\s*Y": "SEVERITY",
        r"D\s*A\s*T\s*E": "DATE",
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
        label
        for label in STOP_LABELS
        if _normalize_label(label) not in excluded
    ]

    return "|".join(re.escape(label) for label in stop_labels)


def _extract_between_labels(text: str, labels: list[str]) -> str:
    """
    Handles continuous OCR text:
    PATIENT NAME Mohamed Rizwan PATIENT ID BT-RAD-5562
    """
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


def _extract_from_line_or_next_line(text: str, labels: list[str]) -> str:
    """
    Handles:
    PATIENT NAME
    Mohamed Rizwan
    """
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


def _extract_value_from_page_text(text: str, field_name: str) -> str:
    labels = FIELD_ALIASES.get(field_name, [])

    if not labels:
        return ""

    return (
        _extract_from_line_or_next_line(text, labels)
        or _extract_between_labels(text, labels)
    )


def _extract_section_from_page_text(text: str, field_name: str) -> str:
    labels = FIELD_ALIASES.get(field_name, [])
    fixed = _normalize_ocr_spacing(text)
    lines = [line.strip() for line in fixed.splitlines() if line.strip()]
    stop_norms = {_normalize_label(label) for label in STOP_LABELS}

    for index, line in enumerate(lines):
        line_norm = _normalize_label(line)

        for label in labels:
            if line_norm == _normalize_label(label):
                collected: list[str] = []

                for next_line in lines[index + 1 :]:
                    next_norm = _normalize_label(next_line)

                    if next_norm in stop_norms:
                        break

                    collected.append(next_line)

                if collected:
                    return " ".join(collected).strip()

    return _extract_between_labels(text, labels)


def _fill_missing_fields_from_page_text(data: dict[str, Any]) -> dict[str, Any]:
    """
    If the AI returns page_text but misses structured fields, extract fields from page_text.
    This is important for direct PNG files and designed report layouts.
    """
    page_text = data.get("page_text", "") or ""

    for field_name in [
        "patient_name",
        "medical_condition",
        "incident_date",
        "hospital_location",
        "severity_level",
    ]:
        if not data.get(field_name):
            value = _extract_value_from_page_text(page_text, field_name)

            if field_name == "severity_level":
                value = _safe_severity(value)

            if value:
                data[field_name] = value

    if not data.get("doctor_notes"):
        notes = _extract_section_from_page_text(page_text, "doctor_notes")
        if notes:
            data["doctor_notes"] = notes

    if not data.get("lab_test_details"):
        lab_details = _extract_section_from_page_text(page_text, "lab_test_details")
        if lab_details:
            data["lab_test_details"] = lab_details

    return data


class AiOcrService:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def process_document(self, file_path: str) -> DocumentOcrResult:
        suffix = Path(file_path).suffix.lower()

        if suffix == ".txt":
            text = Path(file_path).read_text(encoding="utf-8", errors="ignore")

            page = PageOcrResult(
                page_number=1,
                page_text=text,
                confidence=1.0,
                fields={"evidence_type": "typed text"},
                warnings=[],
                metadata={
                    "source": "text",
                    "quality_score": 1.0,
                    "blur_score": None,
                    "review_required": False,
                },
            )

            return DocumentOcrResult(
                full_text=text,
                average_confidence=1.0,
                pages=[page],
                metadata={
                    "source": "text",
                    "warnings": [],
                    "review_required": False,
                    "quality_score": 1.0,
                },
            )

        images = _load_images(file_path)

        if not images:
            raise ValueError("No readable pages/images found in uploaded file.")

        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is required for AI OCR on images and PDFs.")

        page_results: list[PageOcrResult] = []

        for index, image in enumerate(images, start=1):
            page_results.append(self._process_page(image=image, page_number=index))

        full_text = "\n\n".join(
            f"[Page {page.page_number}]\n{page.page_text}"
            for page in page_results
            if page.page_text
        )

        confidence_values = [page.confidence for page in page_results]
        average_confidence = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )

        all_warnings = [
            warning
            for page in page_results
            for warning in page.warnings
        ]

        quality_scores = [
            float(page.metadata.get("quality_score", 0.0))
            for page in page_results
        ]

        average_quality_score = (
            sum(quality_scores) / len(quality_scores)
            if quality_scores
            else 0.0
        )

        review_required = (
            average_confidence < settings.min_ocr_confidence
            or any(page.metadata.get("review_required") for page in page_results)
        )

        if review_required:
            all_warnings.append(
                f"Human review recommended. Average OCR confidence={average_confidence:.2f}, threshold={settings.min_ocr_confidence:.2f}."
            )

        return DocumentOcrResult(
            full_text=full_text,
            average_confidence=round(average_confidence, 3),
            pages=page_results,
            metadata={
                "source": "ai_vision_ocr",
                "page_count": len(page_results),
                "warnings": all_warnings,
                "review_required": review_required,
                "average_quality_score": round(average_quality_score, 3),
                "min_ocr_confidence": settings.min_ocr_confidence,
            },
        )

    def _process_page(self, image: Image.Image, page_number: int) -> PageOcrResult:
        preprocessing_warnings: list[str] = []
        preprocessing_metadata: dict[str, Any] = {}

        if settings.enable_image_preprocessing:
            preprocess_result = preprocess_for_ocr(
                image=image,
                page_number=page_number,
                blur_threshold=settings.blur_threshold,
            )

            image_for_ocr = preprocess_result.image
            preprocessing_warnings = preprocess_result.warnings
            preprocessing_metadata = preprocess_result.metadata
        else:
            image_for_ocr = image
            preprocessing_metadata = {
                "page_number": page_number,
                "quality_score": 1.0,
                "final_blur_score": None,
                "was_cropped": False,
                "was_deskewed": False,
                "deskew_angle": 0.0,
            }

        preprocessed_result = self._run_ai_ocr_on_image(
            image=image_for_ocr,
            page_number=page_number,
            preprocessing_metadata=preprocessing_metadata,
            preprocessing_warnings=preprocessing_warnings,
        )

        if (
            preprocessed_result.confidence < settings.min_ocr_confidence
            or self._has_missing_important_fields(preprocessed_result)
        ):
            original_result = self._run_ai_ocr_on_image(
                image=image,
                page_number=page_number,
                preprocessing_metadata={
                    "page_number": page_number,
                    "quality_score": 1.0,
                    "final_blur_score": None,
                    "was_cropped": False,
                    "was_deskewed": False,
                    "deskew_angle": 0.0,
                    "used_original_retry": True,
                },
                preprocessing_warnings=[
                    "OCR retried with original image because preprocessed OCR had low confidence or missing important fields."
                ],
            )

            return self._choose_better_ocr_result(
                first_result=preprocessed_result,
                second_result=original_result,
            )

        return preprocessed_result

    def _run_ai_ocr_on_image(
        self,
        image: Image.Image,
        page_number: int,
        preprocessing_metadata: dict[str, Any],
        preprocessing_warnings: list[str],
    ) -> PageOcrResult:
        data_url = _image_to_data_url(image)

        prompt = """
You are an AI medical investigation OCR and document understanding system.

Read the uploaded medical investigation page carefully.

Important:
- Extract all visible text.
- Many labels are written in uppercase.
- Some labels and values are in two-column layouts.
- Some labels are on one line and the value is on the next line.
- Do not ignore fields just because they are in a designed card layout.
- If a value is visible near a label, extract it.

Field mapping:
- PATIENT NAME -> patient_name
- PATIENT / PT NAME -> patient_name
- DATE -> incident_date
- HOSPITAL -> hospital_location
- PLACE -> hospital_location
- LOCATION -> hospital_location
- CONDITION -> medical_condition
- DIAGNOSIS -> medical_condition
- SEVERITY -> severity_level
- DOCTOR NOTES -> doctor_notes
- LAB/TEST DETAILS -> lab_test_details
- INVESTIGATION DETAILS -> lab_test_details
- RADIOLOGY FINDINGS -> lab_test_details
- EXAMINATION -> lab_test_details
- PEAK FLOW, OXYGEN SAT, RESP. RATE, BLOOD PRESSURE, TROPONIN I -> lab_test_details

Return ONLY valid JSON:
{
  "page_text": "",
  "confidence": 0.0,
  "detected_handwriting": false,
  "patient_name": "",
  "medical_condition": "",
  "incident_date": "",
  "hospital_location": "",
  "severity_level": "Unknown",
  "doctor_notes": "",
  "lab_test_details": "",
  "evidence_type": "",
  "warnings": []
}

Rules:
- Do not guess missing values.
- If a field is clearly visible, extract it.
- confidence must be between 0 and 1.
- severity_level must be one of: Low, Medium, High, Critical, Unknown.
- Add warnings only when text is unclear or uncertain.
""".strip()

        response = self.client.chat.completions.create(
            model=settings.openai_vision_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful AI OCR and medical document extraction assistant.",
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )

        raw = _chat_response_text(response)
        data = _normalize_ocr_data(_extract_json(raw))
        data = _fill_missing_fields_from_page_text(data)

        quality_score = float(preprocessing_metadata.get("quality_score", 1.0))

        combined_confidence = _combine_confidence(
            ai_confidence=data["confidence"],
            image_quality_score=quality_score,
        )

        warnings: list[str] = []
        warnings.extend(preprocessing_warnings)
        warnings.extend(data["warnings"])

        review_required = combined_confidence < settings.min_ocr_confidence

        if review_required:
            warnings.append(
                f"Low OCR confidence on page {page_number}. Final confidence={combined_confidence:.2f}, threshold={settings.min_ocr_confidence:.2f}."
            )

        fields = {
            "patient_name": data["patient_name"],
            "medical_condition": data["medical_condition"],
            "incident_date": data["incident_date"],
            "hospital_location": data["hospital_location"],
            "severity_level": data["severity_level"],
            "doctor_notes": data["doctor_notes"],
            "lab_test_details": data["lab_test_details"],
            "evidence_type": data["evidence_type"],
        }

        metadata = {
            **preprocessing_metadata,
            "ai_confidence": data["confidence"],
            "final_confidence": combined_confidence,
            "review_required": review_required,
            "detected_handwriting": data["detected_handwriting"],
        }

        return PageOcrResult(
            page_number=page_number,
            page_text=data["page_text"],
            confidence=combined_confidence,
            detected_handwriting=data["detected_handwriting"],
            fields=fields,
            warnings=warnings,
            metadata=metadata,
        )

    def _has_missing_important_fields(self, result: PageOcrResult) -> bool:
        important_fields = [
            "patient_name",
            "medical_condition",
            "incident_date",
            "hospital_location",
        ]

        return any(not result.fields.get(field) for field in important_fields)

    def _field_count(self, result: PageOcrResult) -> int:
        fields = [
            "patient_name",
            "medical_condition",
            "incident_date",
            "hospital_location",
            "severity_level",
            "doctor_notes",
            "lab_test_details",
        ]

        return sum(1 for field in fields if result.fields.get(field))

    def _choose_better_ocr_result(
        self,
        first_result: PageOcrResult,
        second_result: PageOcrResult,
    ) -> PageOcrResult:
        first_score = self._field_count(first_result) + first_result.confidence
        second_score = self._field_count(second_result) + second_result.confidence

        if second_score > first_score:
            second_result.warnings.append(
                "Original image OCR selected because it extracted more fields or had better confidence."
            )
            return second_result

        first_result.warnings.append(
            "Preprocessed image OCR selected because it extracted more fields or had better confidence."
        )
        return first_result


ai_ocr_service = AiOcrService()