"""Пример вопроса через тот же сервис, что и API."""

import os

from studymind.config import get_settings
from studymind.infrastructure.embeddings.factory import build_embedder
from studymind.infrastructure.qdrant_store import build_qdrant
from studymind.rag.service import StudyMindService


QUESTION = "Почему возникает переобучение?"
USER_ID = int(os.getenv("STUDYMIND_USER_ID", "5214477341"))

settings = get_settings()
service = StudyMindService(
    settings=settings,
    qdrant=build_qdrant(settings),
    embedder=build_embedder(settings),
)

answer = service.answer(QUESTION, USER_ID)

print("\nОтвет:")
print(answer.text)
print(f"\nПровайдер: {answer.provider}, режим: {answer.mode}")
print("\nИсточники:")
for source in answer.sources:
    print(f"- {source.document}, страница PDF {source.page}")
