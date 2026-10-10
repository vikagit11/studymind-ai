
from qdrant_client import QdrantClient

from studymind.infrastructure.embeddings.embedding_client import (
    EmbeddingClient,
)


COLLECTION_NAME = "studymind_documents"
QUESTION = "Почему возникает переобучение?"
RELEVANT_PAGES = {44}
K = 5

qdrant = QdrantClient(host="localhost", port=6333)
embedding_client = EmbeddingClient()

query_vector = embedding_client.embed(QUESTION)

results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=K,
).points

# Считаем релевантные результаты в Top-K
relevant_count = sum(
    result.payload["page"] in RELEVANT_PAGES
    for result in results
)

precision_at_k = relevant_count / K

# Ищем позицию первого релевантного результата
reciprocal_rank = 0.0

for rank, result in enumerate(results, start=1):
    if result.payload["page"] in RELEVANT_PAGES:
        reciprocal_rank = 1 / rank
        break

print("Вопрос:", QUESTION)
print("Найдено chunks:", len(results))
print(f"Precision@{K}: {precision_at_k:.3f}")
print(f"Reciprocal Rank: {reciprocal_rank:.3f}")

for rank, result in enumerate(results, start=1):
    print(
        f"Ранг: {rank} | "
        f"Страница: {result.payload['page']} | "
        f"Score: {result.score:.3f}"
    )
