from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Query, Session

from app.models import EvidenceChunk, Investigation
from app.schemas import SearchResult
from app.services.embeddings import embedding_service


def _apply_filters(query: Query, filters: dict[str, Any] | None) -> Query:
    filters = filters or {}
    severity_level = filters.get("severity_level")
    hospital_location = filters.get("hospital_location")
    medical_condition = filters.get("medical_condition")

    if severity_level:
        query = query.filter(Investigation.severity_level.ilike(severity_level))
    if hospital_location:
        query = query.filter(Investigation.hospital_location.ilike(f"%{hospital_location}%"))
    if medical_condition:
        query = query.filter(Investigation.medical_condition.ilike(f"%{medical_condition}%"))
    return query


def _to_result(chunk: EvidenceChunk, investigation: Investigation, score: float, **extra_scores: float | None) -> SearchResult:
    return SearchResult(
        investigation_id=investigation.id,
        chunk_id=chunk.id,
        chunk_index=chunk.chunk_index,
        source_page=chunk.source_page,
        score=round(float(score), 6),
        keyword_score=extra_scores.get("keyword_score"),
        semantic_score=extra_scores.get("semantic_score"),
        content=chunk.content,
        original_filename=investigation.original_filename,
        patient_name=investigation.patient_name,
        medical_condition=investigation.medical_condition,
        hospital_location=investigation.hospital_location,
        severity_level=investigation.severity_level,
        created_at=investigation.created_at,
    )


def keyword_search(db: Session, query_text: str, limit: int = 10, filters: dict[str, Any] | None = None) -> list[SearchResult]:
    ts_vector = func.to_tsvector("english", EvidenceChunk.content)
    ts_query = func.plainto_tsquery("english", query_text)
    rank = func.ts_rank(ts_vector, ts_query)

    query = (
        db.query(EvidenceChunk, Investigation, rank.label("rank"))
        .join(Investigation, EvidenceChunk.investigation_id == Investigation.id)
        .filter(ts_vector.op("@@")(ts_query))
    )
    query = _apply_filters(query, filters)
    rows = query.order_by(rank.desc()).limit(limit).all()
    return [_to_result(chunk, inv, float(rank_value or 0.0), keyword_score=float(rank_value or 0.0)) for chunk, inv, rank_value in rows]


def semantic_search(
    db: Session,
    query_text: str,
    limit: int = 10,
    filters: dict[str, Any] | None = None,
    exclude_investigation_id: str | None = None,
) -> list[SearchResult]:
    query_embedding = embedding_service.embed_text(query_text)
    distance = EvidenceChunk.embedding.cosine_distance(query_embedding)
    score = (1 - distance).label("semantic_score")

    query = (
        db.query(EvidenceChunk, Investigation, score)
        .join(Investigation, EvidenceChunk.investigation_id == Investigation.id)
        .filter(EvidenceChunk.embedding.is_not(None))
    )
    if exclude_investigation_id:
        query = query.filter(Investigation.id != exclude_investigation_id)
    query = _apply_filters(query, filters)
    rows = query.order_by(distance.asc()).limit(limit).all()
    return [_to_result(chunk, inv, float(score_value or 0.0), semantic_score=float(score_value or 0.0)) for chunk, inv, score_value in rows]


def hybrid_search(db: Session, query_text: str, limit: int = 10, filters: dict[str, Any] | None = None) -> list[SearchResult]:
    keyword_results = keyword_search(db, query_text=query_text, limit=limit * 2, filters=filters)
    semantic_results = semantic_search(db, query_text=query_text, limit=limit * 2, filters=filters)

    merged: dict[str, SearchResult] = {}
    max_keyword = max((result.keyword_score or 0.0 for result in keyword_results), default=0.0) or 1.0

    for result in keyword_results:
        result.keyword_score = (result.keyword_score or 0.0) / max_keyword
        result.semantic_score = 0.0
        result.score = round(0.45 * result.keyword_score, 6)
        merged[str(result.chunk_id)] = result

    for result in semantic_results:
        existing = merged.get(str(result.chunk_id))
        semantic_score = max(0.0, min(1.0, result.semantic_score or 0.0))
        if existing:
            existing.semantic_score = semantic_score
            existing.score = round(0.45 * (existing.keyword_score or 0.0) + 0.55 * semantic_score, 6)
        else:
            result.keyword_score = 0.0
            result.semantic_score = semantic_score
            result.score = round(0.55 * semantic_score, 6)
            merged[str(result.chunk_id)] = result

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:limit]
