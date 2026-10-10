import re


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_text(
    text: str,
    chunk_size: int = 500,
    overlap_sentences: int = 1,
) -> list[str]:

    sentences = split_sentences(text)

    chunks = []
    current_sentences = []

    for sentence in sentences:
        current_text = " ".join(current_sentences)

        if len(current_text) + len(sentence) + 1 <= chunk_size:
            current_sentences.append(sentence)
        else:
            if current_sentences:
                chunks.append(" ".join(current_sentences))

            current_sentences = current_sentences[-overlap_sentences:]
            current_sentences.append(sentence)

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks