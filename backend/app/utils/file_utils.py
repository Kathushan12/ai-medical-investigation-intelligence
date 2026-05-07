import mimetypes
import os
import re
import uuid
from pathlib import Path

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".txt"}


def safe_filename(filename: str) -> str:
    stem = Path(filename).stem
    suffix = Path(filename).suffix.lower()
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_") or "evidence"
    return f"{stem}_{uuid.uuid4().hex}{suffix}"


def validate_file_extension(filename: str) -> None:
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}")


def guess_mime_type(path: str) -> str:
    return mimetypes.guess_type(path)[0] or "application/octet-stream"


def ensure_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)
