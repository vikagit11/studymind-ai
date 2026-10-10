
from studymind.rag.pdf_parser import extract_pages


pdf_path = "data/machine_learning.pdf"

pages = extract_pages(pdf_path)

print("Количество страниц с текстом:", len(pages))

for page in pages[:3]:
    print(f"\n--- Страница {page['page']} ---")
    print(page["text"][:500])
