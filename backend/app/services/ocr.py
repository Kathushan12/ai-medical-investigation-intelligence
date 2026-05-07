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
    """
    Works with OpenAI chat.completions.create response.
    """
    try:
        return response.choices[0].message.content or ""
    except Exception:
        return ""


def _normalize_ocr_data(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_text": str(data.get("page_text", "")),
        "confidence": float(data.get("confidence", 0.0) or 0.0),
        "detected_handwriting": bool(data.get("detected_handwriting", False)),
        "patient_name": str(data.get("patient_name", "")),
        "medical_condition": str(data.get("medical_condition", "")),
        "incident_date": str(data.get("incident_date", "")),
        "hospital_location": str(data.get("hospital_location", "")),
        "severity_level": str(data.get("severity_level", "Unknown") or "Unknown"),
        "doctor_notes": str(data.get("doctor_notes", "")),
        "lab_test_details": str(data.get("lab_test_details", "")),
        "evidence_type": str(data.get("evidence_type", "")),
        "warnings": data.get("warnings", []) if isinstance(data.get("warnings", []), list) else [],
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
            return DocumentOcrResult(
                full_text=text,
                average_confidence=1.0,
                pages=[page],
                metadata={"source": "text"},
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

        return DocumentOcrResult(
            full_text=full_text,
            average_confidence=average_confidence,
            pages=page_results,
            metadata={
                "source": "ai_vision_ocr",
                "page_count": len(page_results),
                "warnings": [
                    warning
                    for page in page_results
                    for warning in page.warnings
                ],
            },
        )

    def _process_page(self, image: Image.Image, page_number: int) -> PageOcrResult:
        data_url = _image_to_data_url(image)

        system_prompt = """
You are an AI medical investigation document understanding system.

Your task:
1. Read the uploaded medical investigation page.
2. Extract all readable text.
3. Understand document structure.
4. Handle handwriting, low-quality scans, mixed layouts, tables, stamps, and notes.
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
- Add warnings for unclear handwriting, low-quality images, missing fields, or uncertain extraction.
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
                        {"type": "text", "text": system_prompt},
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

        return PageOcrResult(
            page_number=page_number,
            page_text=data["page_text"],
            confidence=data["confidence"],
            detected_handwriting=data["detected_handwriting"],
            fields=fields,
            warnings=data["warnings"],
        )


ai_ocr_service = AiOcrService()