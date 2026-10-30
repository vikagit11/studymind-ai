import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
)

from studymind.rag.chunker import split_text
from studymind.rag.pdf_parser import extract_pages


COLLECTION_NAME = "studymind_documents"
VECTOR_SIZE = 768


def embed_texts(embedding_client, texts: list[str]) -> list[list[float]]:
    """Берёт пакетные эмбеддинги, если клиент их умеет, иначе считает по одному."""

    if not texts:
        return []

    embed_many = getattr(embedding_client, "embed_many", None)
    if callable(embed_many):
        vectors = embed_many(texts)
    else:
        vectors = [embedding_client.embed(text) for text in texts]

    if len(vectors) != len(texts):
        raise ValueError("Эмбеддер вернул неверное число векторов.")
    return vectors


def index_pdf(
    pdf_path: str,
    qdrant: QdrantClient,
    embedding_client,
    document_id: str,
    user_id: int,
    vector_size: int | None = None,
    collection_name: str = COLLECTION_NAME,
) -> int:
    path = Path(pdf_path)
    expected_size = VECTOR_SIZE if vector_size is None else vector_size

    if not path.is_file():
        raise FileNotFoundError(f"PDF не найден: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Поддерживаются только PDF-файлы.")

    if document_id is None:
        raise ValueError("Передайте постоянный document_id для документа.")

    if not qdrant.collection_exists(collection_name):
        raise ValueError(f"Коллекция {collection_name} не существует.")

    pages = extract_pages(str(path))
    if not pages:
        raise ValueError(f"Не удалось извлечь текст из {path.name}.")

    old_ids = _existing_point_ids(
        qdrant,
        document_id,
        user_id,
        collection_name,
    )
    points = []

    for page in pages:
        chunks = split_text(
            page["text"],
            chunk_size=500,
            overlap_sentences=1,
        )
        vectors = embed_texts(embedding_client, chunks)

        for chunk_index, (chunk, vector) in enumerate(zip(chunks, vectors)):
            if len(vector) != expected_size:
                raise ValueError(
                    f"Размерность embedding: {len(vector)}, "
                    f"ожидалось {expected_size}."
                )

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{document_id}:{page['page']}:{chunk_index}",
                )
            )
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "text": chunk,
                        "source": path.name,
                        "document_id": document_id,
                        "user_id": user_id,
                        "page": page["page"],
                        "chunk_id": chunk_index,
                    },
                )
            )

    qdrant.upsert(
        collection_name=collection_name,
        points=points,
        wait=True,
    )

    new_ids = {point.id for point in points}
    stale_ids = old_ids - new_ids
    if stale_ids:
        qdrant.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=list(stale_ids)),
            wait=True,
        )

    return len(points)


def _existing_point_ids(
    qdrant: QdrantClient,
    document_id: str,
    user_id: int,
    collection_name: str,
) -> set:
    old_ids = set()
    offset = None
    source_filter = Filter(
        must=[
            FieldCondition(
                key="document_id",
                match=MatchValue(value=document_id),
            ),
            FieldCondition(
                key="user_id",
                match=MatchValue(value=user_id),
            ),
        ]
    )

    while True:
        records, offset = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=source_filter,
            limit=500,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        old_ids.update(record.id for record in records)
        if offset is None:
            break

    return old_ids
