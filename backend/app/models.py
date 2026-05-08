import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database import Base


class Investigation(Base):
    __tablename__ = "investigations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=True)
    file_path = Column(Text, nullable=False)
    mime_type = Column(String(100), nullable=True)
    file_size = Column(Integer, nullable=True)

    upload_status = Column(String(50), default="completed", nullable=False)
    ocr_status = Column(String(50), default="pending", nullable=False)
    embedding_status = Column(String(50), default="pending", nullable=False)
    processing_status = Column(String(50), default="queued", nullable=False)

    extracted_text = Column(Text, nullable=True)
    extraction_metadata = Column(JSON, default=dict)

    summary = Column(Text, nullable=True)
    patient_name = Column(String(255), nullable=True)
    medical_condition = Column(String(255), nullable=True)
    incident_date = Column(String(100), nullable=True)
    hospital_location = Column(String(255), nullable=True)
    severity_level = Column(String(50), default="Unknown", nullable=True)
    doctor_notes = Column(Text, nullable=True)
    lab_test_details = Column(Text, nullable=True)

    ocr_confidence = Column(Float, nullable=True)

    # OCR improvement fields
    ocr_warnings = Column(JSON, default=list)
    ocr_metadata = Column(JSON, default=dict)
    extraction_warnings = Column(JSON, default=list)

    # Image preprocessing and quality fields
    document_quality_score = Column(Float, nullable=True)
    blur_score = Column(Float, nullable=True)

    # Human review workflow
    review_required = Column(Boolean, default=False)
    review_reason = Column(Text, nullable=True)

    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
    )

    chunks = relationship(
        "EvidenceChunk",
        back_populates="investigation",
        cascade="all, delete-orphan",
    )


class EvidenceChunk(Base):
    __tablename__ = "evidence_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    investigation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("investigations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    source_page = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)

    # text-embedding-3-small uses 1536 dimensions
    embedding = Column(Vector(1536), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    investigation = relationship(
        "Investigation",
        back_populates="chunks",
    )