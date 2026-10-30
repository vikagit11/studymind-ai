import pytest

from studymind.config import Settings, get_settings


def isolated_settings(**overrides) -> Settings:
    """Настройки для тестов без чтения локального `.env`.

    На машине разработчика в `.env` часто уже лежит OPENAI_API_KEY.
    Без `_env_file=None` pytest начинает зависеть от окружения.
    """

    defaults = {
        "llm_provider": "ollama",
        "embedding_provider": "ollama",
        "openai_api_key": None,
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
