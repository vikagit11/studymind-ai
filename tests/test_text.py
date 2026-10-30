from langchain_core.messages import AIMessage

from studymind.rag.text import is_relevant, message_text, unique_sources


def test_message_text_strips_think_block():
    message = AIMessage(content="<think>черновик</think>\nИтоговый ответ.")
    assert message_text(message) == "Итоговый ответ."


def test_message_text_reads_content_blocks():
    message = AIMessage(content=[{"type": "text", "text": "Ответ"}, " хвост"])
    assert message_text(message) == "Ответ хвост"


def test_relevance_verdicts():
    assert is_relevant("YES") is True
    assert is_relevant("Да, фрагмент полезен") is True
    assert is_relevant("NO.") is False
    assert is_relevant("Нет") is False
    assert is_relevant("не уверен") is None
    assert is_relevant("not relevant") is None


def test_unique_sources_skip_pages_without_number():
    sources = unique_sources(
        [
            {"source": "b.pdf", "page": 2},
            {"source": "a.pdf", "page": 1},
            {"source": "a.pdf", "page": 1},
            {"source": "a.pdf", "page": None},
        ]
    )
    assert sources == [
        {"document": "a.pdf", "page": 1},
        {"document": "b.pdf", "page": 2},
    ]
