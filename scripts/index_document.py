
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from studymind.infrastructure.embeddings.embedding_client import EmbeddingClient
from studymind.rag.document_indexer import index_pdf


COLLECTION_NAME = "studymind_documents"
PDF_PATH = Path("data/machine_learning.pdf")

qdrant = QdrantClient(host="localhost", port=6333)
embedding_client = EmbeddingClient()

# Создаём коллекцию, только если её ещё нет.
# Существующую коллекцию не удаляем.
if not qdrant.collection_exists(COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE,
        ),
    )

# Используем постоянный ID для этой книги.
# При повторном запуске chunks получат те же ID.
document_id = "machine_learning_book_v1"

count = index_pdf(
    pdf_path=str(PDF_PATH),
    qdrant=qdrant,
    embedding_client=embedding_client,
    document_id=document_id,
    user_id=5214477341,
)

print(f"Загружено chunks: {count}")
print("Индексация завершена!")

