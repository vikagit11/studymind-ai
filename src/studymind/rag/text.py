import re
from typing import Any

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_WORD = re.compile(r"[\s,.!:;]+")


def message_text(message: Any) -> str:
    """Достаёт текст из строки или сообщения LangChain.

    У Qwen3 в Ollama в ответ иногда попадает блок `<think>`. Для RAG нужен
    только итоговый текст, поэтому такой блок убирается.
    """

    if isinstance(message, str):
        text = message
    else:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    parts.append(str(block.get("text") or block.get("content") or ""))
                else:
                    parts.append(str(getattr(block, "text", "")))
            text = "".join(parts)
        else:
            text = str(content)

    return _THINK_BLOCK.sub("", text).strip()


def is_relevant(verdict: str) -> bool | None:
    """YES/ДА — фрагмент оставляем, NO/НЕТ — отбрасываем.

    None означает, что модель ответила не по формату. Вызывающий код
    оставляет фрагмент: лучше лишний контекст, чем пустой ответ из-за
    кривого классификатора маленькой локальной модели.
    """

    if not verdict:
        return None

    first = _WORD.split(verdict.strip().upper(), maxsplit=1)[0]
    if first in {"YES", "ДА"}:
        return True
    if first in {"NO", "НЕТ"}:
        return False
    return None


def chunks_as_dicts(chunks) -> list[dict]:
    return [
        {
            "text": chunk.text,
            "source": chunk.source,
            "page": chunk.page,
            "score": chunk.score,
        }
        for chunk in chunks
    ]


def format_context(documents: list[dict]) -> str:
    parts = []
    for document in documents:
        parts.append(
            f"Source: {document.get('source', 'Unknown')}\n"
            f"PDF page: {document.get('page')}\n"
            f"Text:\n{document.get('text', '')}"
        )
    return "\n\n".join(parts)


def unique_sources(documents: list[dict]) -> list[dict]:
    seen: set[tuple[str, int]] = set()
    sources: list[dict] = []

    for document in documents:
        page = document.get("page")
        if not isinstance(page, int):
            continue
        key = (str(document.get("source") or "Unknown"), page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"document": key[0], "page": page})

    sources.sort(key=lambda item: (item["document"], item["page"]))
    return sources
