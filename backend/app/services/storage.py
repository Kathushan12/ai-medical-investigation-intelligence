from pathlib import Path

from fastapi import UploadFile

from app.config import settings
from app.utils.file_utils import ensure_directory, safe_filename, validate_file_extension


def save_upload_file(upload_file: UploadFile) -> tuple[str, str]:
    validate_file_extension(upload_file.filename or "")
    ensure_directory(settings.upload_dir)

    stored_filename = safe_filename(upload_file.filename or "evidence")
    destination = Path(settings.upload_dir) / stored_filename

    with destination.open("wb") as buffer:
        while chunk := upload_file.file.read(1024 * 1024):
            buffer.write(chunk)

    return stored_filename, str(destination)
