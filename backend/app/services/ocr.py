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

    if severity not in {"Low", "Medium", "High", "Critical", "Unknown"}:
        return "Unknown"

    return severity


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
        "patient_name": str(data.get("patient_name", "")),
        "medical_condition": str(data.get("medical_condition", "")),
        "incident_date": str(data.get("incident_date", "")),
        "hospital_location": str(data.get("hospital_location", "")),
        "severity_level": _safe_severity(data.get("severity_level")),
        "doctor_notes": str(data.get("doctor_notes", "")),
        "lab_test_details": str(data.get("lab_test_details", "")),
        "evidence_type": str(data.get("evidence_type", "")),
        "warnings": warnings,
    }


def _combine_confidence(ai_confidence: float, image_quality_score: float) -> float:
    """
    Final confidence combines model confidence and image quality.
    AI confidence is more important, but poor camera image quality should reduce final confidence.
    """
    combined = (ai_confidence * 0.75) + (image_quality_score * 0.25)
    return round(max(0.0, min(1.0, combined)), 3)


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

        data_url = _image_to_data_url(image_for_ocr)

        prompt = """
You are an AI medical investigation document understanding system.

Your task:
1. Read the uploaded medical investigation page.
2. Extract all readable text.
3. Understand the document structure.
4. Handle handwriting, low-quality scans, mixed layouts, tables, stamps, and doctor notes.
5. Extract meaningful medical investigation information.
6. Do not guess missing values.

Return ONLY valid JSON with this exact structure:
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
- confidence must be between 0 and 1.
- severity_level must be one of: Low, Medium, High, Critical, Unknown.
- Use empty string for unavailable fields.
- Add warnings for unclear handwriting, low-quality image, missing fields, or uncertain extraction.
- Never invent patient names, diagnoses, hospitals, dates, or lab results.
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

        quality_score = float(preprocessing_metadata.get("quality_score", 1.0))
        combined_confidence = _combine_confidence(
            ai_confidence=data["confidence"],
            image_quality_score=quality_score,
        )

        warnings = []
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


ai_ocr_service = AiOcrService()