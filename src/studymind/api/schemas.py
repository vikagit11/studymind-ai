from typing import Literal

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    embedding_provider: str
    embedding_dimensions: int
    openai_configured: bool
    rag_mode: str
    qdrant: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    user_id: int = Field(gt=0)
    provider: Literal["openai", "ollama"] | None = None
    mode: Literal["graph", "chain"] | None = None

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        stripped = value.strip()
        if len(stripped) < 3:
            raise ValueError("Вопрос должен содержать минимум 3 символа.")
        return stripped


class Source(BaseModel):
    document: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    provider: str
    mode: str


class UploadResponse(BaseModel):
    message: str
    document_id: str
    chunks: int
