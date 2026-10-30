from studymind.config import get_settings
from studymind.infrastructure.embeddings.factory import build_embedder
from studymind.infrastructure.qdrant_store import build_qdrant


QUESTION = "Почему возникает переобучение?"
RELEVANT_PAGES = {44}
K = 5

settings = get_settings()
qdrant = build_qdrant(settings)
embedder = build_embedder(settings)
query_vector = embedder.embed(QUESTION)

results = qdrant.query_points(
    collection_name=settings.qdrant_collection,
    query=query_vector,
    limit=K,
).points

relevant_count = sum(
    (result.payload or {}).get("page") in RELEVANT_PAGES
    for result in results
)
precision_at_k = relevant_count / K

reciprocal_rank = 0.0
for rank, result in enumerate(results, start=1):
    if (result.payload or {}).get("page") in RELEVANT_PAGES:
        reciprocal_rank = 1 / rank
        break

print("Вопрос:", QUESTION)
print("Найдено chunks:", len(results))
print(f"Precision@{K}: {precision_at_k:.3f}")
print(f"Reciprocal Rank: {reciprocal_rank:.3f}")

for rank, result in enumerate(results, start=1):
    payload = result.payload or {}
    score = result.score or 0.0
    print(
        f"Ранг: {rank} | "
        f"Страница: {payload.get('page')} | "
        f"Score: {score:.3f}"
    )
