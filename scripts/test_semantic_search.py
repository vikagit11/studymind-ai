from studymind.config import get_settings
from studymind.infrastructure.embeddings.factory import build_embedder
from studymind.infrastructure.qdrant_store import build_qdrant


QUESTION = "Что такое Python?"

settings = get_settings()
qdrant = build_qdrant(settings)
embedder = build_embedder(settings)
query_vector = embedder.embed(QUESTION)

results = qdrant.query_points(
    collection_name=settings.qdrant_collection,
    query=query_vector,
    limit=1,
).points

print("Результаты поиска:")

for result in results:
    payload = result.payload or {}
    print("Score:", result.score)
    print("Text:", payload.get("text"))
    print("Source:", payload.get("source"))
