from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    page: int | None
    score: float


@dataclass(frozen=True)
class SourceRef:
    document: str
    page: int


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[SourceRef]
    provider: str
    mode: str
