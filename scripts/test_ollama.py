from studymind.infrastructure.llm.ollama_client import OllamaClient

client = OllamaClient()
answer = client.generate("Объясни простыми словами, что такое RAG")
print(answer)