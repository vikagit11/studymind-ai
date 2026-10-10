
import os
import logging

import httpx
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)


load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
API_URL = "http://127.0.0.1:8000/ask"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)


async def start(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    await update.message.reply_text(
    f"Твой Telegram ID: {update.effective_user.id}"
)
    await update.message.reply_text(
        "Привет! Я StudyMind AI 📚\n\n"
        "Задай вопрос по учебным материалам, "
        "и я постараюсь найти ответ в документах."
    )


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    question = update.message.text.strip()

    if not question:
        await update.message.reply_text("Отправь текстовый вопрос.")
        return

    await update.message.reply_text("Ищу ответ в учебных материалах...")

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
            API_URL,
            json={
            "question": question,
            "user_id": update.effective_user.id,
            },
        )

            response.raise_for_status()
            data = response.json()

        answer = data["answer"]
        sources = data.get("sources", [])

        if sources:
            source_lines = [
                f"• {source['document']}, стр. {source['page']}"
                for source in sources
            ]

            answer += "\n\nИсточники:\n" + "\n".join(source_lines)

        # Telegram ограничивает длину одного сообщения.
        for start_index in range(0, len(answer), 4000):
            await update.message.reply_text(
                answer[start_index:start_index + 4000]
            )

    
    except httpx.HTTPStatusError as exc:
        logger.exception("FastAPI вернул ошибку")

        if exc.response.status_code == 404:
            await update.message.reply_text(
            "В загруженных учебных материалах я не нашёл "
            "достаточно информации, чтобы ответить на этот вопрос."
        )
        else:
            await update.message.reply_text(
            "Не удалось обработать вопрос. Попробуй ещё раз."
        )


    except httpx.RequestError:
        logger.exception("Не удалось подключиться к FastAPI")
        await update.message.reply_text(
            "Не удаётся подключиться к StudyMind AI. "
            "Проверь, запущен ли API-сервер."
        )

    except Exception:
        logger.exception("Ошибка при обработке сообщения")
        await update.message.reply_text(
            "Произошла ошибка при обработке вопроса."
        )


async def handle_pdf(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    message = update.message

    if not message or not message.document:
        return

    document = message.document

    if not document.file_name or not document.file_name.lower().endswith(".pdf"):
        await message.reply_text("Пожалуйста, отправь PDF-файл.")
        return

    if document.file_size and document.file_size > 10 * 1024 * 1024:
        await message.reply_text("Размер PDF не должен превышать 10 МБ.")
        return

    await message.reply_text("Получаю PDF и индексирую документ...")

    try:
        telegram_file = await context.bot.get_file(document.file_id)
        file_bytes = await telegram_file.download_as_bytearray()

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                "http://127.0.0.1:8000/upload",
                data={"user_id": update.effective_user.id},
                files={
                    "file": (
                        document.file_name,
                        bytes(file_bytes),
                        "application/pdf",
                    )
                },
            )
            response.raise_for_status()
            result = response.json()
            print("UPLOAD RESULT:", result)
        await message.reply_text(
            f"PDF успешно обработан!\n"
            f"Фрагментов проиндексировано: {result['chunks']}\n\n"
            "Теперь можешь задавать вопросы по документу."
        )

    except httpx.HTTPStatusError as exc:
        logger.error("Ошибка загрузки PDF: %s", exc.response.text)
        await message.reply_text(
            "Не удалось обработать PDF. Проверь файл и попробуй ещё раз."
        )
    except Exception:
        logger.exception("Ошибка при загрузке PDF")
        await message.reply_text(
            "Не удалось загрузить PDF. Проверь, запущен ли StudyMind AI."
        )

def main() -> None:
    if not TOKEN:
        raise RuntimeError(
            "Не найден TELEGRAM_BOT_TOKEN. Проверь файл .env."
        )

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.add_handler(
    MessageHandler(filters.Document.ALL, handle_pdf)
    )
    logger.info("StudyMind AI Telegram bot запускается")
    application.run_polling()


if __name__ == "__main__":
    main()
