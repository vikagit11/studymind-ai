


from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    SetPayload,
)


COLLECTION_NAME = "studymind_documents"
SOURCE = "machine_learning.pdf"
DOCUMENT_ID = "machine_learning_book_v1"

qdrant = QdrantClient(host="localhost", port=6333)

# Находим старые chunks этой книги.
result_filter = Filter(
    must=[
        FieldCondition(
            key="source",
            match=MatchValue(value=SOURCE),
        )
    ]
)

updated_count = 0
offset = None

while True:
    records, offset = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=result_filter,
        limit=100,
        offset=offset,
        with_payload=False,
        with_vectors=False,
    )

    if records:
        ids = [record.id for record in records]

        qdrant.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"document_id": DOCUMENT_ID},
            points=ids,
            wait=True,
        )

        updated_count += len(ids)

    if offset is None:
        break

print(f"Обновлено точек: {updated_count}")
print(f"document_id: {DOCUMENT_ID}")

