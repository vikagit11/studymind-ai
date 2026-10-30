import logging
from typing import Literal

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from studymind.rag.chain import build_answer_chain
from studymind.rag.graph.state import RAGState
from studymind.rag.prompts import FALLBACK_ANSWER, GRADE_PROMPT, REWRITE_PROMPT
from studymind.rag.text import (
    chunks_as_dicts,
    format_context,
    is_relevant,
    message_text,
    unique_sources,
)

logger = logging.getLogger(__name__)

Route = Literal["generate", "rewrite", "fallback"]


def route_after_grade(state: RAGState, *, max_rewrite_attempts: int) -> Route:
    if state["documents"]:
        return "generate"
    if state["rewrite_count"] < max_rewrite_attempts:
        return "rewrite"
    return "fallback"


def build_rag_graph(llm, retriever, *, max_rewrite_attempts: int):
    """Граф: поиск → оценка фрагментов → ответ, переформулировка или отказ.

    Оценка и переформулировка — это условные шаги. Их неудобно выражать
    одной цепочкой LCEL, поэтому оркестрация живёт в LangGraph, а сами
    вызовы модели остаются цепочками LangChain.
    """

    grade_chain = GRADE_PROMPT | llm | RunnableLambda(message_text)
    rewrite_chain = REWRITE_PROMPT | llm | RunnableLambda(message_text)
    answer_chain = build_answer_chain(llm)

    def retrieve(state: RAGState) -> dict:
        documents = chunks_as_dicts(
            retriever.search(state["question"], state["user_id"])
        )
        return {"documents": documents}

    def grade(state: RAGState) -> dict:
        kept: list[dict] = []
        for document in state["documents"]:
            try:
                verdict = grade_chain.invoke(
                    {
                        "question": state["original_question"],
                        "chunk": document.get("text", ""),
                    }
                )
            except Exception:
                logger.warning(
                    "Грейдер не ответил, фрагмент оставлен в контексте",
                    exc_info=True,
                )
                kept.append(document)
                continue

            decision = is_relevant(verdict)
            if decision is False:
                continue
            kept.append(document)

        logger.info(
            "После оценки осталось %s из %s фрагментов",
            len(kept),
            len(state["documents"]),
        )
        return {"documents": kept}

    def rewrite(state: RAGState) -> dict:
        try:
            rewritten = rewrite_chain.invoke(
                {"question": state["original_question"]}
            ).strip()
        except Exception:
            logger.warning("Не удалось переформулировать вопрос", exc_info=True)
            rewritten = ""

        if not rewritten:
            rewritten = state["question"]

        return {
            "question": rewritten,
            "rewrite_count": state["rewrite_count"] + 1,
        }

    def generate(state: RAGState) -> dict:
        answer = answer_chain.invoke(
            {
                "context": format_context(state["documents"]),
                "question": state["original_question"],
            }
        )
        return {"generation": answer}

    def fallback(state: RAGState) -> dict:
        return {"generation": FALLBACK_ANSWER, "documents": []}

    def route(state: RAGState) -> Route:
        return route_after_grade(
            state,
            max_rewrite_attempts=max_rewrite_attempts,
        )

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade", grade)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.add_node("fallback", fallback)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route,
        {
            "generate": "generate",
            "rewrite": "rewrite",
            "fallback": "fallback",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)
    return graph.compile()


def run_graph(
    llm,
    retriever,
    question: str,
    user_id: int,
    *,
    max_rewrite_attempts: int,
) -> tuple[str, list[dict]]:
    graph = build_rag_graph(
        llm,
        retriever,
        max_rewrite_attempts=max_rewrite_attempts,
    )
    result = graph.invoke(
        {
            "question": question,
            "original_question": question,
            "user_id": user_id,
            "documents": [],
            "generation": "",
            "rewrite_count": 0,
        },
        config={"recursion_limit": 12},
    )
    text = (result.get("generation") or "").strip()
    if not text:
        raise RuntimeError("Модель вернула пустой ответ.")
    return text, unique_sources(result.get("documents") or [])
