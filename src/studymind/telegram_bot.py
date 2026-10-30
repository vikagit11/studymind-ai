import logging

import httpx
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from studymind.config import get_settings

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

PROVIDERS = {"openai", "ollama"}
MODES = {"graph", "chain"}


def _api_url(path: str) -> str:
    base = get_settings().api_base_url.rstrip("/")
    return f"{base}{path}"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if update.message is None or user is None:
        return

    await update.message.reply_text(
        "Привет! Я StudyMind AI.\n\n"
        f"Твой Telegram ID: {user.id}\n\n"
        "Пришли PDF, затем задай вопрос по нему.\n"
        "Команды:\n"
        "/provider openai|ollama|default — какая модель отвечает\n"
        "/mode graph|chain|default — LangGraph или короткая цепочка LangChain\n"
        "/status — текущий выбор\n"
        "/help — то же самое ещё раз"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await start(update, context)


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    settings = get_settings()
    provider = context.user_data.get("provider") or f"{settings.llm_provider} (по умолчанию)"
    mode = context.user_data.get("mode") or f"{settings.rag_mode} (по умолчанию)"
    await update.message.reply_text(
        f"Провайдер ответов: {provider}\n"
        f"Режим: {mode}\n"
        f"Эмбеддинги: {settings.embedding_provider} "
        f"({settings.embedding_dimensions})"
    )


async def set_provider(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    argument = context.args[0].lower() if context.args else ""
    if argument == "default":
        context.user_data.pop("provider", None)
        await update.message.reply_text("Провайдер ответов сброшен на значение из .env.")
        return

    if argument not in PROVIDERS:
        await update.message.reply_text("Использование: /provider openai, /provider ollama или /provider default")
        return

    context.user_data["provider"] = argument
    await update.message.reply_text(f"Провайдер ответов: {argument}")


async def set_mode(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None:
        return

    argument = context.args[0].lower() if context.args else ""
    if argument == "default":
        context.user_data.pop("mode", None)
        await update.message.reply_text("Режим сброшен на значение из .env.")
        return

    if argument not in MODES:
        await update.message.reply_text("Использование: /mode graph, /mode chain или /mode default")
        return

    context.user_data["mode"] = argument
    await update.message.reply_text(f"Режим: {argument}")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message is None or update.effective_user is None:
        return

    question = update.message.text.strip()
    if not question:
        await update.message.reply_text("Отправь текстовый вопрос.")
        return

    provider = context.user_data.get("provider")
    mode = context.user_data.get("mode")
    await update.message.reply_text(
        "Ищу ответ в учебных материалах "
        f"({provider or 'провайдер по умолчанию'}, {mode or 'режим по умолчанию'})..."
    )

    payload = {
        "question": question,
        "user_id": update.effective_user.id,
    }
    if provider:
        payload["provider"] = provider
    if mode:
        payload["mode"] = mode

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(_api_url("/ask"), json=payload)
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

        for start_index in range(0, len(answer), 4000):
            await update.message.reply_text(answer[start_index:start_index + 4000])
    except httpx.HTTPStatusError as exc:
        logger.exception("FastAPI вернул ошибку")
        detail = _error_detail(exc.response)
        if exc.response.status_code == 400 and detail:
            await update.message.reply_text(detail)
        else:
            await update.message.reply_text(
                "Не удалось обработать вопрос. Попробуй ещё раз."
            )
    except httpx.RequestError:
        logger.exception("Не удалось подключиться к FastAPI")
        await update.message.reply_text(
            "Не удаётся подключиться к StudyMind AI. Проверь, запущен ли API-сервер."
        )
    except Exception:
        logger.exception("Ошибка при обработке сообщения")
        await update.message.reply_text("Произошла ошибка при обработке вопроса.")


async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message is None or message.document is None or update.effective_user is None:
        return

    document = message.document
    if not document.file_name or not document.file_name.lower().endswith(".pdf"):
        await message.reply_text("Пожалуйста, отправь PDF-файл.")
        return

    if document.file_size and document.file_size > get_settings().max_upload_bytes:
        await message.reply_text("Размер PDF не должен превышать 10 МБ.")
        return

    await message.reply_text("Получаю PDF и индексирую документ...")

    try:
        telegram_file = await context.bot.get_file(document.file_id)
        file_bytes = await telegram_file.download_as_bytearray()

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                _api_url("/upload"),
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

        await message.reply_text(
            "PDF успешно обработан.\n"
            f"Фрагментов проиндексировано: {result['chunks']}\n\n"
            "Теперь можно задавать вопросы по документу."
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


def _error_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except Exception:
        return ""
    return detail if isinstance(detail, str) else ""


def main() -> None:
    token = get_settings().telegram_bot_token
    if not token:
        raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN. Проверь файл .env.")

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status))
    application.add_handler(CommandHandler("provider", set_provider))
    application.add_handler(CommandHandler("mode", set_mode))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_pdf))

    logger.info("StudyMind AI Telegram bot запускается")
    application.run_polling()


if __name__ == "__main__":
    main()
