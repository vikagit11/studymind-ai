# StudyMind AI

StudyMind AI is a Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from uploaded PDF documents.

## Features

- Upload PDF documents through a FastAPI endpoint.
- Extract and split document text into chunks.
- Generate embeddings using a local embedding model.
- Store and retrieve document chunks using Qdrant.
- Generate answers using a local LLM through Ollama.
- Interact with the application through a Telegram bot.
- Filter retrieved documents by user ID.

## Tech Stack

- **Language:** Python
- **API:** FastAPI, Uvicorn
- **LLM:** Ollama
- **Embeddings:** Ollama embedding model
- **Vector Database:** Qdrant
- **PDF Processing:** pypdf
- **Bot:** python-telegram-bot
- **Configuration:** python-dotenv
- **Containerization:** Docker

## Architecture

1. A user uploads a PDF document.
2. The application extracts the text and splits it into chunks.
3. The embedding model converts chunks into vectors.
4. Qdrant stores the vectors and associated metadata.
5. A user submits a question through the API or Telegram bot.
6. The application retrieves relevant chunks from Qdrant.
7. The local language model generates an answer based on the retrieved context.

## Requirements

- Python 3.13
- Docker
- Ollama
- A compatible LLM and embedding model downloaded through Ollama

## Setup

Clone the repository and navigate to the project directory.

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure the required environment variables in a local `.env` file. Do not commit secrets or API tokens to GitHub.

Start Qdrant and Ollama, ensuring that the application configuration matches their addresses and ports.

Run the API from the project root:

```bash
uvicorn src.studymind.main:app --reload
```

Open the interactive API documentation at:

`http://127.0.0.1:8000/docs`

## Project Status

StudyMind AI is a project demonstrating the implementation of a local RAG pipeline with vector search, a REST API, and Telegram integration.