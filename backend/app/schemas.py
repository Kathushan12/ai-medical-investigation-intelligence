from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field


class InvestigationBase(BaseModel):
    id: UUID
    original_filename: str
    mime_type: str | None = None

    upload_status: str
    ocr_status: str
    embedding_status: str
    processing_status: str

    summary: str | None = None
    patient_name: str | None = None
    medical_condition: str | None = None
    incident_date: str | None = None
    hospital_location: str | None = None
    severity_level: str | None = None
    doctor_notes: str | None = None
    lab_test_details: str | None = None

    ocr_confidence: float | None = None

    # OCR improvement fields
    ocr_warnings: list[str] | None = None
    ocr_metadata: dict[str, Any] | None = None
    extraction_warnings: list[str] | None = None

    # Image quality and preprocessing fields
    document_quality_score: float | None = None
    blur_score: float | None = None

    # Human review fields
    review_required: bool | None = False
    review_reason: str | None = None

    error_message: str | None = None

    created_at: datetime
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class InvestigationDetail(InvestigationBase):
    extracted_text: str | None = None
    extraction_metadata: dict[str, Any] | None = None

    # Optional detailed OCR data for one investigation view
    ocr_text: str | None = None
    ocr_warnings: list[str] | None = None
    ocr_metadata: dict[str, Any] | None = None
    extraction_warnings: list[str] | None = None


class InvestigationStatus(BaseModel):
    id: UUID

    upload_status: str
    ocr_status: str
    embedding_status: str
    processing_status: str

    ocr_confidence: float | None = None
    document_quality_score: float | None = None
    blur_score: float | None = None

    review_required: bool | None = False
    review_reason: str | None = None

    error_message: str | None = None

    model_config = {"from_attributes": True}


class SearchResult(BaseModel):
    investigation_id: UUID
    chunk_id: UUID
    chunk_index: int
    source_page: int | None = None

    score: float
    keyword_score: float | None = None
    semantic_score: float | None = None

    content: str
    original_filename: str

    patient_name: str | None = None
    medical_condition: str | None = None
    hospital_location: str | None = None
    severity_level: str | None = None

    ocr_confidence: float | None = None
    review_required: bool | None = None

    created_at: datetime | None = None


class AssistantQuery(BaseModel):
    question: str = Field(min_length=3)
    top_k: int = Field(default=5, ge=1, le=15)

    severity_level: str | None = None
    hospital_location: str | None = None
    medical_condition: str | None = None


class Citation(BaseModel):
    source_id: str
    investigation_id: UUID
    chunk_id: UUID
    original_filename: str
    chunk_index: int
    source_page: int | None = None
    excerpt: str


class AssistantResponse(BaseModel):
    answer: str
    citations: list[Citation]
    retrieved_results: list[SearchResult]


class UploadResponse(BaseModel):
    message: str
    investigation: InvestigationBase


class HumanReviewSummary(BaseModel):
    investigation_id: UUID
    original_filename: str
    review_required: bool
    review_reason: str | None = None
    ocr_confidence: float | None = None
    document_quality_score: float | None = None
    blur_score: float | None = None
    ocr_warnings: list[str] | None = None
    extraction_warnings: list[str] | None = None

    model_config = {"from_attributes": True}