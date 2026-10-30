import pytest

from studymind.config import Settings
from studymind.errors import ProviderNotConfigured
from studymind.infrastructure.embeddings.factory import build_embedder
from studymind.infrastructure.llm.factory import build_chat_model
from studymind.infrastructure.qdrant_store import vector_size_of


def test_settings_normalize_provider_and_blank_key():
    settings = Settings(
        llm_provider="OpenAI",
        embedding_provider="OLLAMA",
        rag_mode="CHAIN",
        openai_api_key="   ",
    )

    assert settings.llm_provider == "openai"
    assert settings.embedding_provider == "ollama"
    assert settings.rag_mode == "chain"
    assert settings.openai_api_key is None
    assert settings.openai_configured is False
    assert settings.embedding_dimensions == 768


def test_openai_dimensions_follow_embedding_provider():
    settings = Settings(
        embedding_provider="openai",
        openai_api_key="sk-test",
        openai_embedding_dimensions=768,
    )
    assert settings.embedding_dimensions == 768


def test_openai_chat_requires_api_key():
    settings = Settings(llm_provider="openai", openai_api_key=None)
    with pytest.raises(ProviderNotConfigured):
        build_chat_model(settings)


def test_build_ollama_chat_model():
    settings = Settings(llm_provider="ollama", ollama_model="qwen3:1.7b")
    provider, model = build_chat_model(settings, provider="ollama")
    assert provider == "ollama"
    assert model.model == "qwen3:1.7b"


def test_build_openai_chat_model_does_not_call_network():
    settings = Settings(
        llm_provider="openai",
        openai_api_key="sk-test",
        openai_model="gpt-4o-mini",
        openai_base_url="https://example.invalid/v1",
    )
    provider, model = build_chat_model(settings)
    assert provider == "openai"
    assert model.model_name == "gpt-4o-mini"


def test_embedder_factory_selects_dimensions():
    ollama_settings = Settings(embedding_provider="ollama")
    openai_settings = Settings(
        embedding_provider="openai",
        openai_api_key="sk-test",
        openai_embedding_dimensions=768,
    )

    assert build_embedder(ollama_settings).dimensions == 768
    assert build_embedder(openai_settings).dimensions == 768


def test_openai_embedder_requires_api_key():
    settings = Settings(embedding_provider="openai", openai_api_key=None)
    with pytest.raises(ProviderNotConfigured):
        build_embedder(settings)


def test_vector_size_reads_named_and_single_vectors():
    class Vector:
        def __init__(self, size: int) -> None:
            self.size = size

    class Info:
        def __init__(self, vectors) -> None:
            self.config = type("Config", (), {})()
            self.config.params = type("Params", (), {"vectors": vectors})()

    assert vector_size_of(Info(Vector(768))) == 768
    assert vector_size_of(Info({"dense": Vector(1536)})) == 1536
