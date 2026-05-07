from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import Investigation
from app.schemas import InvestigationBase, InvestigationDetail, InvestigationStatus, SearchResult, UploadResponse
from app.services.processing import process_investigation_task
from app.services.related import find_related_investigations
from app.services.storage import save_upload_file
from app.utils.file_utils import guess_mime_type

router = APIRouter(prefix="/investigations", tags=["Investigations"])


@router.post("/upload", response_model=UploadResponse, status_code=201)
def upload_investigation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        stored_filename, file_path = save_upload_file(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    investigation = Investigation(
        original_filename=file.filename or stored_filename,
        stored_filename=stored_filename,
        file_path=file_path,
        mime_type=file.content_type or guess_mime_type(file_path),
        upload_status="uploaded",
        ocr_status="pending",
        embedding_status="pending",
        processing_status="pending",
    )
    db.add(investigation)
    db.commit()
    db.refresh(investigation)

    background_tasks.add_task(process_investigation_task, str(investigation.id))
    return UploadResponse(message="File uploaded. Background processing started.", investigation=investigation)


@router.get("", response_model=list[InvestigationBase])
def list_investigations(db: Session = Depends(get_db)):
    return db.query(Investigation).order_by(Investigation.created_at.desc()).all()


@router.get("/{investigation_id}", response_model=InvestigationDetail)
def get_investigation(investigation_id: UUID, db: Session = Depends(get_db)):
    investigation = db.get(Investigation, investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return investigation


@router.get("/{investigation_id}/status", response_model=InvestigationStatus)
def get_investigation_status(investigation_id: UUID, db: Session = Depends(get_db)):
    investigation = db.get(Investigation, investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return investigation


@router.post("/{investigation_id}/reprocess", response_model=InvestigationStatus)
def reprocess_investigation(
    investigation_id: UUID,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    investigation = db.get(Investigation, investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="Investigation not found")
    investigation.ocr_status = "pending"
    investigation.embedding_status = "pending"
    investigation.processing_status = "pending"
    investigation.error_message = None
    db.commit()
    db.refresh(investigation)
    background_tasks.add_task(process_investigation_task, str(investigation.id))
    return investigation


@router.get("/{investigation_id}/related", response_model=list[SearchResult])
def related_investigations(investigation_id: UUID, limit: int = 5, db: Session = Depends(get_db)):
    investigation = db.get(Investigation, investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="Investigation not found")
    return find_related_investigations(db, investigation, limit=limit)


@router.delete("/{investigation_id}", status_code=204)
def delete_investigation(investigation_id: UUID, db: Session = Depends(get_db)):
    investigation = db.get(Investigation, investigation_id)
    if not investigation:
        raise HTTPException(status_code=404, detail="Investigation not found")
    db.delete(investigation)
    db.commit()
    return None
