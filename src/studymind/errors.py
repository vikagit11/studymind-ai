class StudyMindError(Exception):
    """Базовая ошибка приложения, которую можно показать клиенту."""


class ProviderNotConfigured(StudyMindError):
    """Выбран OpenAI, но в окружении нет ключа."""


class VectorSizeMismatch(StudyMindError):
    """Коллекция Qdrant создана под другую размерность эмбеддингов."""
