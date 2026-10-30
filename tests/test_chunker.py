from studymind.rag.chunker import split_text


def test_short_text_stays_one_chunk():
    text = "Первое предложение. Второе предложение."
    assert split_text(text, chunk_size=500) == [text]


def test_overlap_keeps_previous_sentence():
    sentences = [
        "aaaaaaaaaa.",
        "bbbbbbbbbb.",
        "cccccccccc.",
        "dddddddddd.",
    ]
    chunks = split_text(" ".join(sentences), chunk_size=25, overlap_sentences=1)

    assert chunks == [
        "aaaaaaaaaa. bbbbbbbbbb.",
        "bbbbbbbbbb. cccccccccc.",
        "cccccccccc. dddddddddd.",
    ]


def test_zero_overlap_does_not_repeat_sentences():
    chunks = split_text(
        "aaaaaaaaaa. bbbbbbbbbb. cccccccccc.",
        chunk_size=25,
        overlap_sentences=0,
    )

    assert chunks == [
        "aaaaaaaaaa. bbbbbbbbbb.",
        "cccccccccc.",
    ]


def test_long_sentence_is_split_by_chunk_size():
    sentence = "слово " * 20
    chunks = split_text(sentence.strip() + ".", chunk_size=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 20 for chunk in chunks)


def test_empty_text():
    assert split_text("   ") == []
