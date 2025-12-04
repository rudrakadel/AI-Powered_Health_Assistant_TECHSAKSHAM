# 🏥 Medical AI Assistant

A production-style **medical question‑answering system** built with:

- **FastAPI** backend
- **Celery + Redis** for asynchronous processing
- **RAG** (Retrieval-Augmented Generation) over a Kaggle medical Q&A dataset using **ChromaDB**
- Multiple LLM providers: **Ollama** (local, default), **OpenAI**, and **Google Gemini**
- A clean **Streamlit UI**
- Optional **Docker Compose** setup to run everything with one command

> This project is for **educational and informational** purposes only.  
> It does **not** provide medical advice, diagnosis, or treatment.

---

## ✨ Features

- Ask free‑form medical questions in natural language.
- Choose between:
  - **Ollama** local models (default, e.g. `llama3.2:3b`)
  - **OpenAI** models (with API key)
  - **Google Gemini** models (with API key)
  - **Auto routing** (selects an available provider)
- Optional **RAG (knowledge base)**:
  - Loads a Kaggle medical chatbot dataset (Q/A + tags).
  - Uses `sentence-transformers` embeddings.
  - Indexes in **ChromaDB**.
  - Retrieves top‑K similar Q/A pairs as context for the LLM.
- Asynchronous architecture:
  - FastAPI enqueues queries as Celery tasks.
  - Redis as broker + result backend.
  - UI polls task status and shows progress.
- Observability:
  - `/health` for API, Redis, and Chroma status.
  - `/api/v1/metrics` for basic usage stats (total queries, success rate).

---

## 🧱 Architecture

```

                ┌────────────────────────┐
                │      Streamlit UI      │
                │   (frontend service)   │
                └─────────┬──────────────┘
                          │ HTTP (REST)
                          ▼
                ┌────────────────────────┐
                │        FastAPI         │
                │  /api/v1/query         │
                │  /api/v1/task/{id}     │
                │  /health, /metrics     │
                └─────────┬──────────────┘
                          │ Celery enqueue
                          ▼
                ┌────────────────────────┐
    Redis        │        Celery          │
(broker +      │   query_task worker    │
results)      └─────────┬──────────────┘
│
│ 1) Optional RAG retrieve via ChromaDB
│ 2) LLM generation via Ollama/OpenAI/Gemini
▼
┌────────────────────────┐
│   Answer + metadata    │
│    stored in Redis     │
└────────────────────────┘

```

**RAG pipeline:**

- `MedicalDatasetLoader` → loads `/app/data/train_data_chatbot.csv`.
- `VectorStore` → builds embeddings (e.g. `all-MiniLM-L6-v2`) and stores in **ChromaDB**.
- `RAGRetriever` → given a query, returns top‑K relevant Q&A entries.

---

## 📂 Project Structure

```

medical-ai-assistant/
├── app/
│   ├── main.py                 \# FastAPI application
│   ├── config.py               \# Settings (env-based)
│   ├── models/                 \# Pydantic schemas
│   ├── tasks/
│   │   └── celery_tasks.py     \# Celery task logic
│   ├── rag/
│   │   ├── dataset_loader.py   \# Loads + preprocesses medical dataset
│   │   ├── vector_store.py     \# ChromaDB + embeddings
│   │   └── retriever.py        \# RAGRetriever
│   ├── llm_providers/
│   │   ├── ollama_provider.py  \# Local Ollama LLM client
│   │   ├── openai_provider.py  \# OpenAI client
│   │   └── gemini_provider.py  \# Gemini client
│   ├── agents/
│   │   ├── router.py           \# Provider selection / auto routing
│   │   └── rag_chain.py        \# RAG orchestration
│   └── utils/
│       └── logger.py           \# Structured logging
├── frontend/
│   └── streamlit_app.py        \# Streamlit UI
├── data/
│   └── train_data_chatbot.csv  \# Medical Q\&A dataset (not in repo)
├── celery_worker.py            \# Celery app entry
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env
└── README.md

```

---

## 🔧 Requirements

Python dependencies are in `requirements.txt`, including:

- `fastapi`, `uvicorn`
- `celery`, `redis`
- `streamlit`
- `langchain`, `chromadb`, `sentence-transformers`
- `ollama`, `openai`, `google-generativeai`

External services:

- **Redis 7+**
- **Ollama** running on the host, with at least one model pulled (e.g. `llama3.2:3b`).

---

## ⚙️ Environment Configuration

Create a `.env` file in the project root:

```

API_HOST=0.0.0.0
API_PORT=8000

DATASET_PATH=/app/data/train_data_chatbot.csv
CHROMA_DB_DIR=/app/chroma_db/medical_kb
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_TOP_K=3

REDIS_URL=redis://redis:6379/0

OLLAMA_BASE_URL=http://host.docker.internal:11434
OLLAMA_MODEL=llama3.2:3b

OPENAI_API_KEY=
GEMINI_API_KEY=

LOG_LEVEL=INFO
APP_ENV=development

```

Place your dataset at `data/train_data_chatbot.csv` so that inside the container it appears at `/app/data/train_data_chatbot.csv`.

---

## 🐳 Docker Setup

### Dockerfile

- Based on `python:3.11-slim`
- Installs dependencies from `requirements.txt`
- Copies the project into `/app`
- Default command: run FastAPI with Uvicorn

### docker-compose.yml

Defines four services:

- `redis` – Redis broker/result backend.
- `api` – FastAPI backend (`app.main:app`).
- `worker` – Celery worker (`query_task`).
- `streamlit` – Frontend UI (`frontend/streamlit_app.py`).

---

## 🚀 Running with Docker

Make sure you have:

- Docker and Docker Compose installed.
- Ollama running on the host and model pulled:

```

ollama serve
ollama pull llama3.2:3b

```

Then from the project root:

```

docker compose build
docker compose up

```

Services:

- FastAPI: http://localhost:8000
- Streamlit UI: http://localhost:8501
- Redis: `localhost:6379` (from the `redis` service)

To stop:

```

docker compose down

```

---

## 💬 Usage

1. Open the UI: http://localhost:8501  
2. Enter a question, e.g.:

   > What are the symptoms of diabetes?

3. Choose configuration:
   - Model: **Ollama** (default) or another provider.
   - Toggle **Use knowledge base (RAG)** on/off.
4. Click **Get answer**.
5. The UI shows:
   - Answer text.
   - Model used, latency, tokens.
   - Retrieved knowledge entries (when RAG is enabled).

You can also call the REST API directly:

```

curl -X POST "http://localhost:8000/api/v1/query" ^
-H "Content-Type: application/json" ^
-d "{\"query\": \"What is diabetes?\", \"model_choice\": \"ollama\", \"use_rag\": true}"

```

---

## 🔐 Safety

- Outputs can be inaccurate, biased, or incomplete.
- Never use this system for real medical diagnosis or treatment.
- Always consult a licensed medical professional.

---

## 🛠 Future Improvements

- Replace Streamlit with a React/Next.js frontend.
- Add chat history and authentication.
- Implement streaming responses from Ollama/OpenAI.
- Add admin views for dataset curation and analytics.
