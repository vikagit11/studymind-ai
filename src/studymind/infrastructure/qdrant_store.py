from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from studymind.config import Settings
from studymind.errors import VectorSizeMismatch


def build_qdrant(settings: Settings) -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        timeout=3,
    )


def vector_size_of(info) -> int:
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        if not vectors:
            raise ValueError("У коллекции нет описания векторов.")
        vectors = next(iter(vectors.values()))

    size = getattr(vectors, "size", None)
    if not isinstance(size, int):
        raise ValueError("Не удалось прочитать размерность коллекции Qdrant.")
    return size


def ensure_collection(client: QdrantClient, settings: Settings) -> None:
    """Создаёт коллекцию или проверяет, что её размерность совпадает с эмбеддингами."""

    name = settings.qdrant_collection
    expected = settings.embedding_dimensions

    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=expected,
                distance=Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name=name,
            field_name="user_id",
            field_schema=PayloadSchemaType.INTEGER,
        )
        client.create_payload_index(
            collection_name=name,
            field_name="document_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        return

    actual = vector_size_of(client.get_collection(name))
    if actual != expected:
        raise VectorSizeMismatch(
            f"Коллекция {name} хранит векторы размерности {actual}, "
            f"а текущие эмбеддинги дают {expected}. "
            "Смените QDRANT_COLLECTION или заново проиндексируйте документы. "
            "Смешивать векторы разных моделей в одной коллекции нельзя."
        )
