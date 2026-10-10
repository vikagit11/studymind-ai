def split_text(text: str, chunk_size: int = 200) -> list[str]:
    chunks = []

    for start in range(0, len(text), chunk_size):
        chunk = text[start:start + chunk_size]
        chunks.append(chunk)

    return chunks


with open("data/test_document.txt", "r", encoding="utf-8") as file:
    text = file.read()


chunks = split_text(text)

print("Количество chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(chunk)