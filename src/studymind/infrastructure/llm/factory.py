from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from studymind.config import Settings
from studymind.errors import ProviderNotConfigured


def resolve_provider(settings: Settings, provider: str | None) -> str:
    selected = (provider or settings.llm_provider).strip().lower()
    if selected not in {"openai", "ollama"}:
        raise ValueError("Провайдер LLM должен быть openai или ollama.")
    return selected


def build_chat_model(
    settings: Settings,
    provider: str | None = None,
) -> tuple[str, BaseChatModel]:
    """Собирает чат-модель LangChain для выбранного провайдера.

    Ключ OpenAI не нужен, пока явно не выбран openai. Поэтому локальный
    Ollama остаётся рабочим значением по умолчанию.
    """

    selected = resolve_provider(settings, provider)

    if selected == "openai":
        if not settings.openai_configured:
            raise ProviderNotConfigured(
                "Для OpenAI задайте OPENAI_API_KEY в файле .env."
            )
        kwargs: dict = {
            "model": settings.openai_model,
            "api_key": settings.openai_api_key,
            "temperature": settings.llm_temperature,
        }
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        return selected, ChatOpenAI(**kwargs)

    return selected, ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.llm_temperature,
    )
