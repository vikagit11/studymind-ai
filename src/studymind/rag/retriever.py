import logging

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from studymind.rag.models import RetrievedChunk

logger = logging.getLogger(__name__)


class QdrantRetriever:
    def __init__(
        self,
        qdrant: QdrantClient,
        embedder,
        collection_name: str,
        top_k: int,
        score_threshold: float,
    ) -> None:
        self._qdrant = qdrant
        self._embedder = embedder
        self._collection = collection_name
        self._top_k = top_k
        self._score_threshold = score_threshold

    def search(self, question: str, user_id: int) -> list[RetrievedChunk]:
        vector = self._embedder.embed(question)
        points = self._qdrant.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="user_id",
                        match=MatchValue(value=user_id),
                    )
                ]
            ),
            limit=self._top_k,
        ).points

        chunks: list[RetrievedChunk] = []
        for point in points:
            score = point.score if isinstance(point.score, (int, float)) else 0.0
            if score < self._score_threshold:
                continue

            payload = point.payload or {}
            text = payload.get("text")
            if not isinstance(text, str) or not text.strip():
                continue

            page = payload.get("page")
            chunks.append(
                RetrievedChunk(
                    text=text,
                    source=str(payload.get("source") or "Unknown"),
                    page=page if isinstance(page, int) else None,
                    score=float(score),
                )
            )

        logger.info(
            "Поиск user_id=%s: кандидатов %s, после порога %s",
            user_id,
            len(points),
            len(chunks),
        )
        return chunks
