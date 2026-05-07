from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas import SearchResult
from app.services.search import hybrid_search, keyword_search, semantic_search

router = APIRouter(prefix="/search", tags=["Search"])


def _filters(severity_level: str | None, hospital_location: str | None, medical_condition: str | None) -> dict:
    return {
        "severity_level": severity_level,
        "hospital_location": hospital_location,
        "medical_condition": medical_condition,
    }


@router.get("/keyword", response_model=list[SearchResult])
def keyword(
    q: str = Query(..., min_length=2),
    limit: int = Query(10, ge=1, le=50),
    severity_level: str | None = None,
    hospital_location: str | None = None,
    medical_condition: str | None = None,
    db: Session = Depends(get_db),
):
    return keyword_search(db, q, limit=limit, filters=_filters(severity_level, hospital_location, medical_condition))


@router.get("/semantic", response_model=list[SearchResult])
def semantic(
    q: str = Query(..., min_length=2),
    limit: int = Query(10, ge=1, le=50),
    severity_level: str | None = None,
    hospital_location: str | None = None,
    medical_condition: str | None = None,
    db: Session = Depends(get_db),
):
    return semantic_search(db, q, limit=limit, filters=_filters(severity_level, hospital_location, medical_condition))


@router.get("/hybrid", response_model=list[SearchResult])
def hybrid(
    q: str = Query(..., min_length=2),
    limit: int = Query(10, ge=1, le=50),
    severity_level: str | None = None,
    hospital_location: str | None = None,
    medical_condition: str | None = None,
    db: Session = Depends(get_db),
):
    return hybrid_search(db, q, limit=limit, filters=_filters(severity_level, hospital_location, medical_condition))
