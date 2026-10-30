import requests


class EmbeddingClient:
    """Прямой клиент Ollama для ручных проверок в `scripts/test_embeddings.py`.

    Приложение использует `build_embedder`: там тот же Ollama или OpenAI
    уже через LangChain.
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url

    def embed(
        self,
        text: str,
        model: str = "embeddinggemma",
    ) -> list[float]:
        response = requests.post(
            f"{self.base_url}/api/embed",
            json={
                "model": model,
                "input": text,
            },
        )

        response.raise_for_status()

        data = response.json()

        return data["embeddings"][0]