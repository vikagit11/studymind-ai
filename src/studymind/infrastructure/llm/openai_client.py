import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class OpenAIClient:
    def __init__(self):
        load_dotenv(PROJECT_ROOT / ".env")

        api_key = os.getenv("OPENAI_API_KEY")
        model_name = os.getenv("OPENAI_MODEL")

        if not api_key:
            raise ValueError(
                "Добавь OPENAI_API_KEY в .env в корне проекта."
            )

        if not model_name:
            raise ValueError(
                "Добавь OPENAI_MODEL в .env в корне проекта."
            )

        self.model = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            timeout=60,
            max_retries=2,
        )

    def generate(self, prompt: str) -> str:
        response = self.model.invoke(prompt)
        answer = response.text.strip()

        if not answer:
            raise ValueError("Модель вернула пустой ответ.")

        return answer
