# StudyMind AI

StudyMind AI отвечает на вопросы по загруженным PDF. Поиск фрагментов идёт через Qdrant, а текст ответа пишет языковая модель: локальная Ollama или OpenAI. Переключается только модель ответа. Эмбеддинги остаются теми, которыми проиндексирована коллекция.

## Что умеет

- Загрузка PDF через API или Telegram.
- Нарезка текста на чанки, эмбеддинги и запись в Qdrant с фильтром по `user_id`.
- Два провайдера ответов: `ollama` и `openai`. Провайдер задаётся в `.env` и может быть переопределён в одном запросе.
- Два режима RAG:
  - `chain` — короткая цепочка LangChain: поиск, промпт, модель.
  - `graph` — граф LangGraph: поиск, оценка фрагментов, ответ, переформулировка вопроса или честный отказ.
- Проверка размерности коллекции, чтобы векторы разных моделей не смешивались.

## Стек

- Python, FastAPI, Pydantic Settings
- LangChain (`ChatOpenAI`, `ChatOllama`, промпты, LCEL)
- LangGraph
- Qdrant, pypdf
- python-telegram-bot

## Как устроен ответ

```text
вопрос
  -> эмбеддинг вопроса
  -> Qdrant, фильтр user_id, порог score
  -> mode=chain: промпт | модель | текст
  -> mode=graph:
       retrieve -> grade -> generate
                         -> rewrite -> retrieve
                         -> fallback
```

Оценка фрагмента — отдельный вызов модели, которая отвечает `YES` или `NO`. Если подходящих фрагментов нет, граф один раз переписывает вопрос и ищет снова. Если контекста всё ещё нет, ответ — прямой отказ.

Если грейдер ответил не `YES`/`NO`, фрагмент остаётся. Для маленькой локальной модели это спокойнее, чем пустой контекст из-за кривого классификатора.

Эмбеддинги и модель ответа разделены. `gpt-4o-mini` и `qwen3` можно менять на каждый вопрос: они не пишут векторы. `embeddinggemma` и `text-embedding-3-small` писать в одну коллекцию нельзя, у них разная размерность. Для OpenAI-эмбеддингов заведите отдельное значение `QDRANT_COLLECTION` и загрузите PDF заново.

## Настройка

Нужны Python 3.11+, Qdrant и, для локальной модели, Ollama. OpenAI нужен только если выбран этот провайдер.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
cp .env.example .env
```

Qdrant:

```bash
docker compose up -d
```

Минимальный локальный `.env`:

```env
LLM_PROVIDER=ollama
EMBEDDING_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_EMBEDDING_MODEL=embeddinggemma
QDRANT_HOST=localhost
QDRANT_PORT=6333
TELEGRAM_BOT_TOKEN=
```

Ответы через OpenAI, эмбеддинги по-прежнему локальные:

```env
LLM_PROVIDER=openai
EMBEDDING_PROVIDER=ollama
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
```

`OPENAI_BASE_URL` нужен только для совместимого шлюза. Пустое значение означает обычный API OpenAI.

Ключ не коммитится: `.env` уже в `.gitignore`. В репозитории лежит только `.env.example`.

## Запуск

API:

```bash
uvicorn studymind.main:app --reload
```

Документация: `http://127.0.0.1:8000/docs`.

Проверка зависимостей: `GET /health`. Если Qdrant недоступен, API поднимется, а `qdrant` в ответе будет `unavailable`.

Telegram-бот, вторым процессом:

```bash
python -m studymind.telegram_bot
```

Без editable-установки те же команды запускаются с `PYTHONPATH=src`.

Команды бота: `/provider openai`, `/provider ollama`, `/mode graph`, `/mode chain`, `/status`.

В запросе к API те же переключатели:

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Почему возникает переобучение?","user_id":1,"provider":"openai","mode":"graph"}'
```

`user_id` обязателен: поиск возвращает только чанки этого пользователя. В Telegram подставляется id пользователя.

## Тесты

```bash
python -m pytest
```

Тесты не ходят в OpenAI, Ollama и Qdrant. Модель и поиск подменяются: проверяются маршруты графа, отказ без контекста, выбор провайдера и ответы API.

Скрипты в `scripts/` — ручные проверки по уже загруженному PDF. `scripts/test_ollama.py` и `scripts/test_embeddings.py` бьют в Ollama напрямую. Остальные берут провайдер из `.env`.

## Что здесь уровня middle

- Настройки в одном `Settings`, а не размазаны по `os.getenv` и константам.
- Фабрика чат-модели: один интерфейс LangChain, две реализации.
- Режим `chain` показывает LCEL. Режим `graph` показывает условный поток, который цепочкой уже неудобно читать.
- Состояние графа типизировано. Узлы возвращают только свой кусок состояния.
- Индексация и поиск используют один эмбеддер. При смене размерности коллекция не перезаписывается молча.
- `create_app()` позволяет подменить сервис в тестах.
- Ошибки провайдера — это 400 с понятным текстом. Падение Qdrant или модели — 503 без стека наружу.

Следующий слой того же проекта: rerank поверх top-k, прогон `scripts/evaluate_retrieval.py` на фиксированном наборе вопросов, асинхронная индексация и кэш эмбеддингов повторяющихся запросов.
