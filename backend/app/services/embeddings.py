import hashlib
import math
import random

from openai import OpenAI

from app.config import settings


def _development_hash_embedding(text: str) -> list[float]:
    """Deterministic fallback so local tests can run without an API key.

    Do not use this for final evaluation. Add OPENAI_API_KEY in .env for real semantic search.
    """
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)
    vector = [rng.uniform(-1.0, 1.0) for _ in range(settings.embedding_dimensions)]
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class EmbeddingService:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

    def embed_text(self, text: str) -> list[float]:
        cleaned = text.strip()
        if not cleaned:
            return [0.0] * settings.embedding_dimensions

        if not self.client:
            return _development_hash_embedding(cleaned)

        response = self.client.embeddings.create(
            model=settings.openai_embedding_model,
            input=cleaned,
        )
        return response.data[0].embedding


embedding_service = EmbeddingService()
