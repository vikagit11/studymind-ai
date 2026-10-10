from qdrant_client import QdrantClient

from studymind.infrastructure.embeddings.embedding_client import EmbeddingClient


COLLECTION_NAME = "studymind_documents"


qdrant = QdrantClient(
    host="localhost",
    port=6333,
)

embedding_client = EmbeddingClient()


question = "Почему возникает переобучение?"


query_vector = embedding_client.embed(question)


results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=3,
).points


print("Вопрос:", question)
print("\nНайденные chunks:")


for result in results:
    print("\nScore:", result.score)
    print("Страница:", result.payload["page"])
    print("Текст:", result.payload["text"])
    print("Источник:", result.payload["source"])
