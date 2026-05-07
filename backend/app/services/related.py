from sqlalchemy.orm import Session

from app.models import Investigation
from app.schemas import SearchResult
from app.services.search import semantic_search


def find_related_investigations(db: Session, investigation: Investigation, limit: int = 5) -> list[SearchResult]:
    query_parts = [
        investigation.patient_name or "",
        investigation.medical_condition or "",
        investigation.hospital_location or "",
        investigation.severity_level or "",
        investigation.summary or "",
        investigation.doctor_notes or "",
        investigation.lab_test_details or "",
    ]
    query_text = " ".join(part for part in query_parts if part).strip()
    if not query_text:
        query_text = investigation.extracted_text or ""
    if not query_text:
        return []
    return semantic_search(
        db=db,
        query_text=query_text,
        limit=limit,
        exclude_investigation_id=str(investigation.id),
    )
