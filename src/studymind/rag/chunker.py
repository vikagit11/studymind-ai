import re


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def split_long_sentence(sentence: str, chunk_size: int) -> list[str]:
    """Режет предложение, которое само длиннее лимита чанка."""

    if len(sentence) <= chunk_size:
        return [sentence]

    parts: list[str] = []
    current = ""

    for word in sentence.split():
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            parts.append(current)
            current = ""

        if len(word) <= chunk_size:
            current = word
            continue

        for start in range(0, len(word), chunk_size):
            parts.append(word[start:start + chunk_size])

    if current:
        parts.append(current)

    return parts


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap_sentences: int = 1,
) -> list[str]:
    sentences: list[str] = []
    for sentence in split_sentences(text):
        sentences.extend(split_long_sentence(sentence, chunk_size))

    chunks: list[str] = []
    current_sentences: list[str] = []

    for sentence in sentences:
        current_text = " ".join(current_sentences)
        extra = len(sentence) if not current_text else len(sentence) + 1

        if not current_sentences or len(current_text) + extra <= chunk_size:
            current_sentences.append(sentence)
            continue

        chunks.append(current_text)
        overlap = (
            current_sentences[-overlap_sentences:]
            if overlap_sentences > 0
            else []
        )
        current_sentences = []
        for previous in reversed(overlap):
            candidate = " ".join([previous, *current_sentences, sentence])
            if len(candidate) > chunk_size:
                break
            current_sentences.insert(0, previous)
        current_sentences.append(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks
