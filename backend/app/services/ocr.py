import base64
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from openai import OpenAI
from PIL import Image

from app.config import settings


@dataclass
class PageOcrResult:
    page_number: int
    page_text: str
    confidence: float
    detected_handwriting: bool = False
    fields: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


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


OCR_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "page_text": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "detected_handwriting": {"type": "boolean"},
        "patient_name": {"type": "string"},
        "medical_condition": {"type": "string"},
        "incident_date": {"type": "string"},
        "hospital_location": {"type": "string"},
        "severity_level": {"type": "string", "enum": ["Low", "Medium", "High", "Critical", "Unknown"]},
        "doctor_notes": {"type": "string"},
        "lab_test_details": {"type": "string"},
        "evidence_type": {"type": "string"},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "page_text",
        "confidence",
        "detected_handwriting",
        "patient_name",
        "medical_condition",
        "incident_date",
        "hospital_location",
        "severity_level",
        "doctor_notes",
        "lab_test_details",
        "evidence_type",
        "warnings",
    ],
}


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
            )
            return DocumentOcrResult(full_text=text, average_confidence=1.0, pages=[page], metadata={"source": "text"})

        images = _load_images(file_path)
        if not images:
            raise ValueError("No readable pages/images found in uploaded file.")

        if not self.client:
            raise RuntimeError("OPENAI_API_KEY is required for AI OCR on images and PDFs.")

        page_results: list[PageOcrResult] = []
        for index, image in enumerate(images, start=1):
            page_results.append(self._process_page(image=image, page_number=index))

        full_text = "\n\n".join(f"[Page {p.page_number}]\n{p.page_text}" for p in page_results if p.page_text)
        confidence_values = [p.confidence for p in page_results]
        average_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        return DocumentOcrResult(
            full_text=full_text,
            average_confidence=average_confidence,
            pages=page_results,
            metadata={
                "source": "ai_vision_ocr",
                "page_count": len(page_results),
                "warnings": [warning for page in page_results for warning in page.warnings],
            },
        )

    def _process_page(self, image: Image.Image, page_number: int) -> PageOcrResult:
        data_url = _image_to_data_url(image)
        prompt = (
            "You are an AI medical investigation document understanding system. "
            "Extract readable text and meaningful investigation information from this page. "
            "Handle handwriting, low-quality scans, tables, stamps, and mixed layouts. "
            "Do not invent missing data. Use empty strings for unavailable fields. "
            "Return a confidence from 0 to 1 and warnings for uncertain extraction."
        )
        response = self.client.responses.create(
            model=settings.openai_vision_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "medical_ocr_page",
                    "strict": True,
                    "schema": OCR_JSON_SCHEMA,
                }
            },
        )
        raw = _response_text(response)
        data = json.loads(raw)
        fields = {
            "patient_name": data.get("patient_name", ""),
            "medical_condition": data.get("medical_condition", ""),
            "incident_date": data.get("incident_date", ""),
            "hospital_location": data.get("hospital_location", ""),
            "severity_level": data.get("severity_level", "Unknown"),
            "doctor_notes": data.get("doctor_notes", ""),
            "lab_test_details": data.get("lab_test_details", ""),
            "evidence_type": data.get("evidence_type", ""),
        }
        return PageOcrResult(
            page_number=page_number,
            page_text=data.get("page_text", ""),
            confidence=float(data.get("confidence") or 0.0),
            detected_handwriting=bool(data.get("detected_handwriting")),
            fields=fields,
            warnings=data.get("warnings", []),
        )


ai_ocr_service = AiOcrService()
