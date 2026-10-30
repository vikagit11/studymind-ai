from fastapi.testclient import TestClient

from studymind.errors import ProviderNotConfigured
from tests.conftest import isolated_settings
from studymind.main import create_app
from studymind.rag.models import Answer, SourceRef


class FakeService:
    def __init__(self, answer: Answer | None = None, error: Exception | None = None):
        self.answer_result = answer or Answer(
            text="Готовый ответ",
            sources=[SourceRef(document="notes.pdf", page=2)],
            provider="ollama",
            mode="graph",
        )
        self.error = error
        self.calls: list[dict] = []

    def answer(self, question: str, user_id: int, provider=None, mode=None) -> Answer:
        self.calls.append(
            {
                "question": question,
                "user_id": user_id,
                "provider": provider,
                "mode": mode,
            }
        )
        if self.error:
            raise self.error
        return self.answer_result


def _client(service: FakeService, settings=None) -> TestClient:
    app = create_app(settings=settings or isolated_settings(), service=service)
    return TestClient(app)


def test_health_reports_provider_configuration():
    response = _client(FakeService()).get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["llm_provider"] == "ollama"
    assert body["embedding_provider"] == "ollama"
    assert body["embedding_dimensions"] == 768
    assert body["openai_configured"] is False
    assert body["qdrant"] == "skipped"


def test_ask_passes_provider_and_mode():
    service = FakeService()
    response = _client(service).post(
        "/ask",
        json={
            "question": "  Что такое RAG?  ",
            "user_id": 5,
            "provider": "openai",
            "mode": "chain",
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "Готовый ответ"
    assert response.json()["provider"] == "ollama"
    assert response.json()["sources"] == [{"document": "notes.pdf", "page": 2}]
    assert service.calls == [
        {
            "question": "Что такое RAG?",
            "user_id": 5,
            "provider": "openai",
            "mode": "chain",
        }
    ]


def test_ask_rejects_short_question():
    response = _client(FakeService()).post(
        "/ask",
        json={"question": "  а ", "user_id": 1},
    )
    assert response.status_code == 422


def test_missing_openai_key_is_client_error():
    service = FakeService(
        error=ProviderNotConfigured("Для OpenAI задайте OPENAI_API_KEY в файле .env.")
    )
    response = _client(service).post(
        "/ask",
        json={"question": "Что такое RAG?", "user_id": 1, "provider": "openai"},
    )

    assert response.status_code == 400
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_dependency_failure_is_unavailable():
    response = _client(FakeService(error=RuntimeError("qdrant down"))).post(
        "/ask",
        json={"question": "Что такое RAG?", "user_id": 1},
    )
    assert response.status_code == 503
    assert "qdrant down" not in response.json()["detail"]
