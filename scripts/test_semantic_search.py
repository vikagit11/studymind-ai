from qdrant_client import QdrantClient

from studymind.infrastructure.embeddings.embedding_client import EmbeddingClient


COLLECTION_NAME = "studymind_documents"


qdrant = QdrantClient(
    host="localhost",
    port=6333,
)

embedding_client = EmbeddingClient()


# Вопрос пользователя
question = "Что такое Python?"


# Превращаем вопрос в embedding
query_vector = embedding_client.embed(question)


# Ищем похожий вектор
results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=1,
).points


print("Результаты поиска:")

for result in results:
    print("Score:", result.score)
    print("Text:", result.payload["text"])
    print("Source:", result.payload["source"])