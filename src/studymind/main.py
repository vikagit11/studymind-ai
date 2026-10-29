import os
from pathlib import Path
from uuid import uuid4
from studymind.rag.document_indexer import index_pdf
from fastapi import UploadFile, File, Form
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue
from studymind.infrastructure.embeddings.embedding_client import EmbeddingClient
from studymind.infrastructure.llm.openai_client import OpenAIClient


COLLECTION_NAME = "studymind_documents"

app = FastAPI(
    title="StudyMind AI",
    description="AI study assistant with RAG",
    version="0.1.0",
)

qdrant = QdrantClient(host="localhost", port=6333)
embedding_client = EmbeddingClient()
llm = OpenAIClient()


class HealthResponse(BaseModel):
    status: str



class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    user_id: int = Field(gt=0)



class Source(BaseModel):
    document: str
    page: int


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def root():
    return {
        "app": "StudyMind AI",
        "message": "AI study assistant is running",
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    try:
        query_vector = embedding_client.embed(question)

        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=Filter(
            must=[
            FieldCondition(
                key="user_id",
                match=MatchValue(value=request.user_id),
            )
        ]
    ),
            limit=3,
        ).points
        print("Найденные фрагменты:", [(r.score, r.payload.get("source"), r.payload.get("text", "")[:150]) for r in results])
        # Оставляем только достаточно релевантные фрагменты.
        results = [
            result
            for result in results
            if result.score >= 0.35
        ]
        if not results:
            raise HTTPException(
                status_code=404,
                detail="No relevant document chunks found.",
            )

        context_parts = []
        sources = set()

        for result in results:
            payload = result.payload or {}

            text = payload.get("text")
            if not text:
                continue

            document = payload.get("source", "Unknown")
            page = payload.get("page")

            context_parts.append(
                f"Source: {document}\n"
                f"PDF page: {page}\n"
                f"Text:\n{text}"
            )

            if isinstance(page, int):
                sources.add((document, page))

        if not context_parts:
            raise HTTPException(
                status_code=404,
                detail="No usable text found in search results.",
            )

        
        prompt = f"""Ты — StudyMind AI, русскоязычный помощник по учебным материалам.

Правила ответа:
1. Всегда отвечай на русском языке, независимо от языка контекста.
2. Отвечай на вопрос, используя предоставленный контекст.
3. Не выдумывай факты и не приписывай контексту информацию, которой в нём нет.
4. Если контекст не содержит достаточно информации для ответа, прямо скажи по-русски:
   "В загруженных учебных материалах я не нашёл достаточно информации, чтобы ответить на этот вопрос."
5. Не пытайся отвечать на вопросы, не связанные с найденным контекстом.
6. Не добавляй список источников в текст ответа — источники возвращаются отдельно.

        Контекст:
        {"\n\n".join(context_parts)}

        Вопрос:
        {question}
"""


        answer = llm.generate(prompt)

        return AskResponse(
            answer=answer,
            sources=[
                Source(document=document, page=page)
                for document, page in sorted(sources)
            ],
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Не удалось обработать вопрос. Проверь доступность Qdrant, "
                "сервиса эмбеддингов и OpenAI API."
            ),
        ) from exc
UPLOAD_DIR = Path("data/uploads")
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: int = Form(...),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Поддерживаются только PDF-файлы.",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    document_id = str(uuid4())
    file_path = UPLOAD_DIR / f"{user_id}_{document_id}.pdf"

    try:
        content = await file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Размер PDF не должен превышать 10 МБ.",
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Файл пуст.",
            )

        file_path.write_bytes(content)

        count = index_pdf(
            pdf_path=str(file_path),
            qdrant=qdrant,
            embedding_client=embedding_client,
            document_id=document_id,
            user_id=user_id,
        )

        return {
            "message": "PDF успешно загружен и проиндексирован.",
            "document_id": document_id,
            "chunks": count,
        }

    except HTTPException:
        raise
    except Exception as exc:
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=500,
            detail="Не удалось обработать PDF.",
        ) from exc
    finally:
        await file.close()
