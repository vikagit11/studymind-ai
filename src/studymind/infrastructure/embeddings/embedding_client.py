import requests


class EmbeddingClient:
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