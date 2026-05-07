from app.services.chunking import chunk_text, normalize_text


def test_normalize_text_removes_extra_spaces():
    assert normalize_text("Hello     world\n\n\nTest") == "Hello world\n\nTest"


def test_chunk_text_returns_chunks():
    text = "Paragraph one.\n\n" + "A" * 1200
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
