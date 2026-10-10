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

from studymind.infrastructure.embeddings.embedding_client import EmbeddingClient
from studymind.rag.chunker import split_text
from studymind.rag.pdf_parser import extract_pages


COLLECTION_NAME = "studymind_documents"
VECTOR_SIZE = 768


def index_pdf(
    pdf_path: str,
    qdrant: QdrantClient,
    embedding_client: EmbeddingClient,
    document_id: str,
    user_id: int,
) -> int:
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF не найден: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Поддерживаются только PDF-файлы.")

    if document_id is None:
        raise ValueError(
            "Передайте постоянный document_id для документа."
        )

    if not qdrant.collection_exists(COLLECTION_NAME):
        raise ValueError(
            f"Коллекция {COLLECTION_NAME} не существует."
        )

    pages = extract_pages(str(path))

    if not pages:
        raise ValueError(
            f"Не удалось извлечь текст из {path.name}."
        )

    # Запоминаем ID уже сохранённых chunks этой книги.
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
            collection_name=COLLECTION_NAME,
            scroll_filter=source_filter,
            limit=500,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )

        old_ids.update(record.id for record in records)

        if offset is None:
            break

    points = []

    for page in pages:
        chunks = split_text(
            page["text"],
            chunk_size=500,
            overlap_sentences=1,
        )

        for chunk_index, chunk in enumerate(chunks):
            vector = embedding_client.embed(chunk)

            if len(vector) != VECTOR_SIZE:
                raise ValueError(
                    f"Размерность embedding: {len(vector)}, "
                    f"ожидалось {VECTOR_SIZE}."
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

    # Сначала записываем новые chunks.
    qdrant.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    # Удаляем только устаревшие ID этой книги.
    new_ids = {point.id for point in points}
    stale_ids = old_ids - new_ids

    if stale_ids:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(
                points=list(stale_ids)
            ),
            wait=True,
        )

    return len(points)
