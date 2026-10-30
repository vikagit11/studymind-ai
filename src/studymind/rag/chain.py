from langchain_core.runnables import RunnableLambda

from studymind.rag.prompts import ANSWER_PROMPT, FALLBACK_ANSWER
from studymind.rag.text import (
    chunks_as_dicts,
    format_context,
    message_text,
    unique_sources,
)


def build_answer_chain(llm):
    """Короткий RAG без циклов: промпт, модель, текст ответа."""

    return ANSWER_PROMPT | llm | RunnableLambda(message_text)


def run_chain(llm, retriever, question: str, user_id: int) -> tuple[str, list[dict]]:
    documents = chunks_as_dicts(retriever.search(question, user_id))
    if not documents:
        return FALLBACK_ANSWER, []

    answer = build_answer_chain(llm).invoke(
        {
            "context": format_context(documents),
            "question": question,
        }
    )
    if not answer:
        raise RuntimeError("Модель вернула пустой ответ.")
    return answer, unique_sources(documents)
