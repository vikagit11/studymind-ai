import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from studymind.api.routes import router
from studymind.config import Settings, get_settings
from studymind.infrastructure.embeddings.factory import build_embedder
from studymind.infrastructure.qdrant_store import build_qdrant, ensure_collection
from studymind.rag.service import StudyMindService

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "service", None) is not None:
        yield
        return

    settings: Settings = app.state.settings
    qdrant = build_qdrant(settings)
    embedder = build_embedder(settings)
    app.state.qdrant = qdrant
    app.state.embedder = embedder

    try:
        ensure_collection(qdrant, settings)
        app.state.qdrant_status = "ok"
    except Exception as exc:
        app.state.qdrant_status = "unavailable"
        logger.warning("Qdrant недоступен при старте: %s", exc)

    app.state.service = StudyMindService(
        settings=settings,
        qdrant=qdrant,
        embedder=embedder,
    )
    logger.info(
        "StudyMind запущен: llm=%s embeddings=%s mode=%s",
        settings.llm_provider,
        settings.embedding_provider,
        settings.rag_mode,
    )
    yield


def create_app(
    settings: Settings | None = None,
    service=None,
) -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="StudyMind AI",
        description=(
            "Учебный RAG-ассистент. Ответы можно получать через OpenAI "
            "или локальную Ollama. Режим graph использует LangGraph, "
            "режим chain — цепочку LangChain."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )
    app.state.settings = settings or get_settings()
    app.state.service = service
    app.state.qdrant_status = "skipped" if service is not None else "unknown"
    app.include_router(router)

    @app.middleware("http")
    async def log_requests(request, call_next):
        response = await call_next(request)
        logger.info(
            "%s %s -> %s",
            request.method,
            request.url.path,
            response.status_code,
        )
        return response

    return app


app = create_app()
