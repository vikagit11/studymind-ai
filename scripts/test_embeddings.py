from studymind.infrastructure.embeddings.embedding_client import EmbeddingClient


client = EmbeddingClient()

vector = client.embed(
    "Python — это язык программирования."
)

print("Размер вектора:", len(vector))
print("Первые 10 значений:", vector[:10])