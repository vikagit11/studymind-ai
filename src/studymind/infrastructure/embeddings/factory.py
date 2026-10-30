from langchain_ollama import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings

from studymind.config import Settings
from studymind.errors import ProviderNotConfigured


class LangChainEmbedder:
    """Адаптер эмбеддингов LangChain к методу `embed`, который ждёт индексатор."""

    def __init__(self, embeddings, dimensions: int) -> None:
        self._embeddings = embeddings
        self.dimensions = dimensions

    def embed(self, text: str) -> list[float]:
        vector = self._embeddings.embed_query(text)
        self._check(vector)
        return list(vector)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._embeddings.embed_documents(list(texts))
        checked = [list(vector) for vector in vectors]
        for vector in checked:
            self._check(vector)
        if len(checked) != len(texts):
            raise ValueError("Эмбеддер вернул неверное число векторов.")
        return checked

    def _check(self, vector: list[float]) -> None:
        if len(vector) != self.dimensions:
            raise ValueError(
                f"Размерность embedding: {len(vector)}, "
                f"ожидалось {self.dimensions}."
            )


def build_embedder(settings: Settings) -> LangChainEmbedder:
    if settings.embedding_provider == "openai":
        if not settings.openai_configured:
            raise ProviderNotConfigured(
                "Для эмбеддингов OpenAI задайте OPENAI_API_KEY в файле .env."
            )
        kwargs: dict = {
            "model": settings.openai_embedding_model,
            "api_key": settings.openai_api_key,
            "dimensions": settings.openai_embedding_dimensions,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        embeddings = OpenAIEmbeddings(**kwargs)
        return LangChainEmbedder(
            embeddings,
            settings.openai_embedding_dimensions,
        )

    if settings.embedding_provider != "ollama":
        raise ValueError("Провайдер эмбеддингов должен быть openai или ollama.")

    embeddings = OllamaEmbeddings(
        model=settings.ollama_embedding_model,
        base_url=settings.ollama_base_url,
    )
    return LangChainEmbedder(
        embeddings,
        settings.ollama_embedding_dimensions,
    )
