from qdrant_client import QdrantClient

from studymind.infrastructure.embeddings.embedding_client import EmbeddingClient
from studymind.infrastructure.llm.ollama_client import OllamaClient


COLLECTION_NAME = "studymind_documents"


qdrant = QdrantClient(
    host="localhost",
    port=6333,
)

embedding_client = EmbeddingClient()
llm = OllamaClient()


question = "Почему возникает переобучение?"


# 1. Превращаем вопрос в embedding
query_vector = embedding_client.embed(question)


# 2. Ищем релевантные chunks
results = qdrant.query_points(
    collection_name=COLLECTION_NAME,
    query=query_vector,
    limit=2,
).points




# 3. Собираем контекст и источники
context_parts = []
sources = set()

for result in results:
    payload = result.payload

    context_parts.append(
        f"""
Источник: {payload["source"]}
Страница PDF: {payload["page"]}
Фрагмент:
{payload["text"]}
"""
    )

    sources.add(
        (payload["source"], payload["page"])
    )

context = "\n\n".join(context_parts)


# 4. Формируем prompt
prompt = f"""
Ты — AI-помощник по обучению.

Ответь на вопрос, используя предоставленный контекст.

Правила:
1. Объясняй понятно и последовательно.
2. Не добавляй факты, которых нет в контексте.
3. Если информации недостаточно, честно скажи об этом.
4. Не формируй список источников самостоятельно.

Контекст:
{context}

Вопрос:
{question}
"""


# 5. Генерируем ответ
answer = llm.generate(prompt)

print("\nОтвет:")
print(answer)

print("\nИсточники найденных фрагментов:")

for source, page in sorted(sources):
    print(f"- {source}, страница PDF {page}")

