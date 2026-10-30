import os
from pathlib import Path

from studymind.config import get_settings
from studymind.infrastructure.embeddings.factory import build_embedder
from studymind.infrastructure.qdrant_store import build_qdrant
from studymind.rag.service import StudyMindService


PDF_PATH = Path("data/machine_learning.pdf")
DOCUMENT_ID = "machine_learning_book_v1"
USER_ID = int(os.getenv("STUDYMIND_USER_ID", "5214477341"))

settings = get_settings()
service = StudyMindService(
    settings=settings,
    qdrant=build_qdrant(settings),
    embedder=build_embedder(settings),
)

count = service.index_document(
    pdf_path=str(PDF_PATH),
    document_id=DOCUMENT_ID,
    user_id=USER_ID,
)

print(f"Загружено chunks: {count}")
print(f"Коллекция: {settings.qdrant_collection}")
print(f"Эмбеддинги: {settings.embedding_provider} ({settings.embedding_dimensions})")
