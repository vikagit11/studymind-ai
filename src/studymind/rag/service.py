from studymind.config import Settings
from studymind.infrastructure.llm.factory import build_chat_model
from studymind.infrastructure.qdrant_store import ensure_collection
from studymind.rag.chain import run_chain
from studymind.rag.document_indexer import index_pdf
from studymind.rag.graph.builder import run_graph
from studymind.rag.models import Answer, SourceRef
from studymind.rag.retriever import QdrantRetriever


class StudyMindService:
    def __init__(
        self,
        settings: Settings,
        qdrant,
        embedder,
        retriever=None,
    ) -> None:
        self.settings = settings
        self.qdrant = qdrant
        self.embedder = embedder
        self.retriever = retriever or QdrantRetriever(
            qdrant=qdrant,
            embedder=embedder,
            collection_name=settings.qdrant_collection,
            top_k=settings.retrieval_top_k,
            score_threshold=settings.retrieval_score_threshold,
        )

    def answer(
        self,
        question: str,
        user_id: int,
        provider: str | None = None,
        mode: str | None = None,
    ) -> Answer:
        resolved_mode = (mode or self.settings.rag_mode).strip().lower()
        if resolved_mode not in {"graph", "chain"}:
            raise ValueError("Режим RAG должен быть graph или chain.")

        selected, llm = build_chat_model(self.settings, provider)

        if resolved_mode == "chain":
            text, sources = run_chain(llm, self.retriever, question, user_id)
        else:
            text, sources = run_graph(
                llm,
                self.retriever,
                question,
                user_id,
                max_rewrite_attempts=self.settings.max_rewrite_attempts,
            )

        return Answer(
            text=text,
            sources=[
                SourceRef(document=item["document"], page=item["page"])
                for item in sources
            ],
            provider=selected,
            mode=resolved_mode,
        )

    def index_document(self, pdf_path: str, document_id: str, user_id: int) -> int:
        if self.qdrant is None or self.embedder is None:
            raise RuntimeError("Индексация недоступна: нет Qdrant или эмбеддера.")

        ensure_collection(self.qdrant, self.settings)
        return index_pdf(
            pdf_path=pdf_path,
            qdrant=self.qdrant,
            embedding_client=self.embedder,
            document_id=document_id,
            user_id=user_id,
            vector_size=self.settings.embedding_dimensions,
            collection_name=self.settings.qdrant_collection,
        )
