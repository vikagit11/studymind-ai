from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ProviderName = Literal["openai", "ollama"]
RagMode = Literal["graph", "chain"]


class Settings(BaseSettings):
    """Настройки читаются из окружения и локального файла `.env`.

    Провайдер ответов и провайдер эмбеддингов разделены специально.
    Векторы в Qdrant зависят только от модели эмбеддингов: GPT и Ollama
    можно переключать на каждый вопрос, а эмбеддинги — нет, пока коллекция
    не переиндексирована.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: ProviderName = "ollama"
    embedding_provider: ProviderName = "ollama"

    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimensions: int = 1536

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"
    ollama_embedding_model: str = "embeddinggemma"
    ollama_embedding_dimensions: int = 768

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "studymind_documents"

    retrieval_top_k: int = 4
    retrieval_score_threshold: float = 0.35
    max_rewrite_attempts: int = 1
    llm_temperature: float = 0.1
    rag_mode: RagMode = "graph"

    api_base_url: str = "http://127.0.0.1:8000"
    telegram_bot_token: str | None = None
    max_upload_bytes: int = 10 * 1024 * 1024

    @field_validator("llm_provider", "embedding_provider", "rag_mode", mode="before")
    @classmethod
    def normalize_choice(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("openai_api_key", "openai_base_url", mode="before")
    @classmethod
    def blank_string_is_missing(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @property
    def embedding_dimensions(self) -> int:
        if self.embedding_provider == "openai":
            return self.openai_embedding_dimensions
        return self.ollama_embedding_dimensions

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
