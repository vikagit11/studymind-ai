from studymind.config import get_settings
from studymind.infrastructure.embeddings.factory import build_embedder
from studymind.infrastructure.qdrant_store import build_qdrant


QUESTION = "Почему возникает переобучение?"

settings = get_settings()
qdrant = build_qdrant(settings)
embedder = build_embedder(settings)
query_vector = embedder.embed(QUESTION)

results = qdrant.query_points(
    collection_name=settings.qdrant_collection,
    query=query_vector,
    limit=3,
).points

print("Вопрос:", QUESTION)
print("\nНайденные chunks:")

for result in results:
    payload = result.payload or {}
    print("\nScore:", result.score)
    print("Страница:", payload.get("page"))
    print("Текст:", payload.get("text"))
    print("Источник:", payload.get("source"))
