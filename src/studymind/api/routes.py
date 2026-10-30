import logging
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile

from studymind.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    Source,
    UploadResponse,
)
from studymind.config import Settings
from studymind.errors import ProviderNotConfigured, VectorSizeMismatch
from studymind.rag.service import StudyMindService

logger = logging.getLogger(__name__)
router = APIRouter()
UPLOAD_DIR = Path("data/uploads")


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_service(request: Request) -> StudyMindService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Сервис не инициализирован.",
        )
    return service


@router.get("/")
def root():
    return {
        "app": "StudyMind AI",
        "message": "AI study assistant is running",
    }


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request, settings: Settings = Depends(get_settings)):
    qdrant_status = getattr(request.app.state, "qdrant_status", "unknown")
    status = "ok" if qdrant_status == "ok" else "degraded"
    return HealthResponse(
        status=status,
        llm_provider=settings.llm_provider,
        embedding_provider=settings.embedding_provider,
        embedding_dimensions=settings.embedding_dimensions,
        openai_configured=settings.openai_configured,
        rag_mode=settings.rag_mode,
        qdrant=qdrant_status,
    )


@router.post("/ask", response_model=AskResponse)
def ask_question(
    body: AskRequest,
    service: StudyMindService = Depends(get_service),
):
    try:
        result = service.answer(
            question=body.question,
            user_id=body.user_id,
            provider=body.provider,
            mode=body.mode,
        )
    except ProviderNotConfigured as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except VectorSizeMismatch as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Не удалось ответить на вопрос")
        raise HTTPException(
            status_code=503,
            detail=(
                "Не удалось обработать вопрос. "
                "Проверьте провайдер LLM, Ollama и Qdrant."
            ),
        ) from exc

    return AskResponse(
        answer=result.text,
        sources=[
            Source(document=item.document, page=item.page)
            for item in result.sources
        ],
        provider=result.provider,
        mode=result.mode,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: int = Form(...),
    service: StudyMindService = Depends(get_service),
    settings: Settings = Depends(get_settings),
):
    if user_id <= 0:
        raise HTTPException(status_code=400, detail="user_id должен быть больше 0.")

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только PDF-файлы.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    document_id = str(uuid4())
    file_path = UPLOAD_DIR / f"{user_id}_{document_id}.pdf"

    try:
        content = await file.read(settings.max_upload_bytes + 1)

        if len(content) > settings.max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail="Размер PDF не должен превышать 10 МБ.",
            )

        if not content:
            raise HTTPException(status_code=400, detail="Файл пуст.")

        file_path.write_bytes(content)
        count = service.index_document(
            pdf_path=str(file_path),
            document_id=document_id,
            user_id=user_id,
        )
        return UploadResponse(
            message="PDF успешно загружен и проиндексирован.",
            document_id=document_id,
            chunks=count,
        )
    except HTTPException:
        if file_path.exists():
            file_path.unlink()
        raise
    except VectorSizeMismatch as exc:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Не удалось обработать PDF")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail="Не удалось обработать PDF.",
        ) from exc
    finally:
        await file.close()
