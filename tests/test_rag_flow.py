import pytest
from langchain_core.runnables import RunnableLambda

from studymind.config import Settings
from studymind.rag.chain import run_chain
from studymind.rag.document_indexer import embed_texts
from studymind.rag.graph.builder import route_after_grade, run_graph
from studymind.rag.models import RetrievedChunk
from studymind.rag.prompts import FALLBACK_ANSWER
from studymind.rag.service import StudyMindService


def chunk(text: str, page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(text=text, source="notes.pdf", page=page, score=0.8)


class ScriptedRetriever:
    def __init__(self, batches: list[list[RetrievedChunk]]) -> None:
        self.batches = [list(batch) for batch in batches]
        self.questions: list[str] = []

    def search(self, question: str, user_id: int) -> list[RetrievedChunk]:
        self.questions.append(question)
        if not self.batches:
            return []
        return self.batches.pop(0)


def scripted_llm(replies: list[str]):
    queue = list(replies)

    def answer(_prompt):
        if not queue:
            raise AssertionError("Модель вызвана больше раз, чем ожидалось")
        return queue.pop(0)

    return RunnableLambda(answer)


def test_route_generates_when_documents_remain():
    state = {
        "question": "q",
        "original_question": "q",
        "user_id": 1,
        "documents": [{"text": "контекст"}],
        "generation": "",
        "rewrite_count": 0,
    }
    assert route_after_grade(state, max_rewrite_attempts=1) == "generate"


def test_route_rewrites_then_falls_back():
    empty = {
        "question": "q",
        "original_question": "q",
        "user_id": 1,
        "documents": [],
        "generation": "",
        "rewrite_count": 0,
    }
    assert route_after_grade(empty, max_rewrite_attempts=1) == "rewrite"
    empty["rewrite_count"] = 1
    assert route_after_grade(empty, max_rewrite_attempts=1) == "fallback"


def test_graph_answers_from_relevant_chunk():
    retriever = ScriptedRetriever([[chunk("Градиент уменьшает ошибку.")]])
    text, sources = run_graph(
        scripted_llm(["YES", "Градиентный спуск уменьшает ошибку."]),
        retriever,
        "Что делает градиентный спуск?",
        user_id=7,
        max_rewrite_attempts=1,
    )

    assert text == "Градиентный спуск уменьшает ошибку."
    assert sources == [{"document": "notes.pdf", "page": 1}]
    assert retriever.questions == ["Что делает градиентный спуск?"]


def test_graph_rewrites_query_when_first_chunks_are_irrelevant():
    retriever = ScriptedRetriever(
        [
            [chunk("Оглавление книги.")],
            [chunk("Переобучение — это запоминание тренировочной выборки.", page=4)],
        ]
    )
    text, sources = run_graph(
        scripted_llm(
            [
                "NO",
                "Почему модель запоминает обучающую выборку?",
                "YES",
                "Модель запоминает тренировочные примеры.",
            ]
        ),
        retriever,
        "Почему возникает переобучение?",
        user_id=7,
        max_rewrite_attempts=1,
    )

    assert retriever.questions == [
        "Почему возникает переобучение?",
        "Почему модель запоминает обучающую выборку?",
    ]
    assert text == "Модель запоминает тренировочные примеры."
    assert sources == [{"document": "notes.pdf", "page": 4}]


def test_graph_rewrites_when_first_search_is_empty():
    retriever = ScriptedRetriever([[], [chunk("Нужный фрагмент.")]])
    text, sources = run_graph(
        scripted_llm(["Новая формулировка", "YES", "Ответ из документа."]),
        retriever,
        "Как работает внимание?",
        user_id=7,
        max_rewrite_attempts=1,
    )

    assert retriever.questions[1] == "Новая формулировка"
    assert text == "Ответ из документа."
    assert sources[0]["page"] == 1


def test_graph_fallback_without_relevant_context():
    retriever = ScriptedRetriever([[chunk("Совсем другой текст.")], []])
    text, sources = run_graph(
        scripted_llm(["NO", "Другая формулировка"]),
        retriever,
        "Как устроен attention?",
        user_id=7,
        max_rewrite_attempts=1,
    )

    assert text == FALLBACK_ANSWER
    assert sources == []


def test_chain_returns_fallback_without_calling_model():
    retriever = ScriptedRetriever([[]])

    def unused(_prompt):
        raise AssertionError("Цепочка не должна вызывать модель без контекста")

    text, sources = run_chain(
        RunnableLambda(unused),
        retriever,
        "Что такое RAG?",
        user_id=7,
    )

    assert text == FALLBACK_ANSWER
    assert sources == []


def test_service_chain_mode_uses_selected_provider(monkeypatch):
    retriever = ScriptedRetriever([[chunk("RAG соединяет поиск и генерацию.")]])
    monkeypatch.setattr(
        "studymind.rag.service.build_chat_model",
        lambda settings, provider=None: ("openai", scripted_llm(["Ответ про RAG."])),
    )
    service = StudyMindService(
        settings=Settings(rag_mode="graph"),
        qdrant=None,
        embedder=None,
        retriever=retriever,
    )

    answer = service.answer(
        "Что такое RAG?",
        user_id=3,
        provider="openai",
        mode="chain",
    )

    assert answer.text == "Ответ про RAG."
    assert answer.provider == "openai"
    assert answer.mode == "chain"
    assert answer.sources[0].page == 1


def test_embed_texts_prefers_batch_method():
    class BatchEmbedder:
        def embed(self, text: str) -> list[float]:
            raise AssertionError("Должен использоваться embed_many")

        def embed_many(self, texts: list[str]) -> list[list[float]]:
            return [[float(len(text))] for text in texts]

    assert embed_texts(BatchEmbedder(), ["ab", "abcd"]) == [[2.0], [4.0]]


def test_embed_texts_falls_back_to_single_calls():
    class SingleEmbedder:
        def embed(self, text: str) -> list[float]:
            return [1.0, 2.0]

    assert embed_texts(SingleEmbedder(), ["a"]) == [[1.0, 2.0]]


def test_unknown_mode_is_rejected():
    service = StudyMindService(
        settings=Settings(),
        qdrant=None,
        embedder=None,
        retriever=ScriptedRetriever([[]]),
    )
    with pytest.raises(ValueError):
        service.answer("Что такое RAG?", user_id=1, mode="agent")
