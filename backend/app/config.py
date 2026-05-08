from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Medical Investigation Intelligence API"
    api_prefix: str = "/api/v1"
    database_url: str = Field(default="postgresql+psycopg2://postgres:postgres@localhost:5432/medical_ai")

    openai_api_key: str | None = None
    openai_vision_model: str = "gpt-4.1-mini"
    openai_text_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    upload_dir: str = "storage/uploads"
    max_pdf_pages: int = 10
    chunk_size: int = 900
    chunk_overlap: int = 150
    cors_origins: str = "http://localhost:3000"

    min_ocr_confidence: float = 0.65
    blur_threshold: float = 80.0
    enable_image_preprocessing: bool = True

    @property
    def cors_origin_list(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
