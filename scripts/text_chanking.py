from studymind.rag.chunker import split_text


with open("data/test_document.txt", "r", encoding="utf-8") as file:
    text = file.read()


chunks = split_text(text, chunk_size=200, overlap_sentences=1)


print("Количество chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(chunk)
    print("Длина:", len(chunk))