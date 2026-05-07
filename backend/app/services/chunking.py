import re

from app.config import settings


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[str]:
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap
    text = normalize_text(text)
    if not text:
        return []

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)
            if len(paragraph) > chunk_size:
                start = 0
                while start < len(paragraph):
                    end = start + chunk_size
                    chunks.append(paragraph[start:end].strip())
                    start = max(end - overlap, end)
            else:
                current = paragraph

    if current:
        chunks.append(current)

    # Add small overlap by prepending the end of previous chunk.
    overlapped: list[str] = []
    for index, chunk in enumerate(chunks):
        if index == 0 or overlap <= 0:
            overlapped.append(chunk)
            continue
        prefix = chunks[index - 1][-overlap:]
        overlapped.append(f"{prefix}\n{chunk}".strip())
    return overlapped
