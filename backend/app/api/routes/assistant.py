from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import AssistantQuery, AssistantResponse
from app.services.assistant import assistant_service

router = APIRouter(prefix="/assistant", tags=["AI Assistant"])


@router.post("/query", response_model=AssistantResponse)
def ask_assistant(payload: AssistantQuery, db: Session = Depends(get_db)):
    filters = {
        "severity_level": payload.severity_level,
        "hospital_location": payload.hospital_location,
        "medical_condition": payload.medical_condition,
    }
    return assistant_service.answer(db, question=payload.question, top_k=payload.top_k, filters=filters)
