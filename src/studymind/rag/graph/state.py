from typing import TypedDict


class RAGState(TypedDict):
    question: str
    original_question: str
    user_id: int
    documents: list[dict]
    generation: str
    rewrite_count: int
