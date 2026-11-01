<div align="center">

# CampusMind - AI Campus Assistant

**A college assistant that answers from official documents with citations, and acts through agents that plan study schedules, set reminders and summarize notes.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agents-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-Strict-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20Search-C52B9B?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![Celery](https://img.shields.io/badge/Celery-Jobs-37814A?style=for-the-badge&logo=celery&logoColor=white)](https://docs.celeryq.dev/)
[![CI](https://img.shields.io/github/actions/workflow/status/Shashank17singh/CampusMind/ci.yml?branch=main&style=for-the-badge&logo=githubactions&logoColor=white&label=CI)](https://github.com/Shashank17singh/CampusMind/actions/workflows/ci.yml)

</div>

---

## Overview

CampusMind gives a college one place to ask about its own rules. Faculty upload circulars, syllabi and notices, and students get answers drawn only from those documents, with numbered sources they can open. If the documents do not support an answer, it says so.

A LangGraph router also sends each message to the right agent. One builds a study plan around exam dates, one sets reminders from plain English, and one summarizes long documents in four formats. Students, faculty and admins each get their own view, and every permission is enforced on the server.

> All tests pass in CI, and the whole app has run in a real browser against real PostgreSQL, Chroma and Ollama. No container has been started, nothing has been deployed, and there are no evaluation results yet. See [Known Limitations](#known-limitations).

---

## Architecture Overview

```mermaid
graph TD
    subgraph "Browser"
    A[React App]
    end

    subgraph "Application"
    B[Nginx] --> C(FastAPI)
    C --> D{LangGraph Router}
    D --> E[RAG Agent]
    D --> F[Planner Agent]
    D --> G[Reminder Agent]
    D --> H[Summarizer Agent]
    end

    subgraph "Retrieval"
    I(Hybrid Search - Vector + BM25) --> J[(Chroma or Qdrant)]
    end

    subgraph "Background Jobs"
    K(Celery Worker and Beat)
    end

    subgraph "Data"
    L[(PostgreSQL 16)]
    M[(Redis - Broker, Cache, Limits)]
    end

    N[Hosted LLM or Ollama]

    A -->|HTTPS, SSE, WebSocket| B
    E --> I
    F --> I
    H --> I
    E --> N
    F --> N
    G --> N
    H --> N
    I --> L
    C --> L
    C --> M
    M -->|Jobs| K
    K --> L
    K --> J
    K -->|Notifications| M

    classDef io fill:#f9f0ff,stroke:#8a2be2,stroke-width:2px,color:#000;
    classDef core fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef logic fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;

    class A,B,N io;
    class C,I,J,K,L,M core;
    class D,E,F,G,H logic;
```

- A question streams back over SSE. The router picks the agent, the RAG agent searches with the student's department and semester as filters, and citations arrive as the final event.
- An upload is stored and queued. A Celery worker extracts, cleans, chunks and embeds it, and reports progress over a WebSocket.
- Celery beat checks for due reminders every minute. An idempotency key stops the same notification being claimed twice.

---

## Features

|                        |                                                                                                                  |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Cited Answers**      | Answers stream token by token, with numbered sources that open beside the answer                                 |
| **Hybrid Search**      | Vector and BM25 results merged with reciprocal rank fusion, plus an optional cross-encoder re-ranker             |
| **Scoped Results**     | Filters by department and semester, while college-wide documents stay visible to everyone                        |
| **Guardrails**         | Refuses when the documents do not support an answer, neutralizes prompt injection, and redacts phones and emails |
| **Agent Graph**        | LangGraph routing with validated tool calls, step limits, timeouts and a stored trace of every node              |
| **Study Planner**      | Scheduled in code, with spaced revision, re-planning and calendar export                                         |
| **Reminders**          | Set from plain English, plus faculty assignments, snooze, and delivery by WebSocket and email                    |
| **Summarizer**         | Map-reduce summaries as brief, detailed, exam notes or flashcards, checked against the source                    |
| **Document Ingestion** | PDF, DOCX, PPTX, TXT and images, with OCR for scans, automatic tagging and duplicate detection                   |
| **Accounts and Roles** | Student, faculty and admin roles, Argon2 passwords, rotating refresh tokens, and login lockout                   |
| **Conversations**      | History, full-text search, feedback on answers, and a retention limit                                            |
| **Admin Analytics**    | Top questions, unanswered questions, knowledge gaps, LLM cost and document status                                |
| **Swappable Parts**    | Chroma or Qdrant, and Anthropic or Ollama, each chosen by one environment variable                               |
| **Evaluation Suite**   | Retrieval benchmarks, an LLM judge, Cohen's kappa against human labels, and a paired bootstrap                   |
| **Web App**            | Light and dark themes, role-based pages, and layouts that hold down to 360px                                     |
| **Monitoring**         | Prometheus metrics and a Grafana dashboard                                                                       |

---

## Tech Stack

| Component      | Technology                                                      |
| -------------- | --------------------------------------------------------------- |
| Backend        | Python 3.11, FastAPI, Pydantic v2, SQLAlchemy 2, Alembic        |
| AI             | LangGraph, sentence-transformers (bge-small-en-v1.5), rank-bm25 |
| LLM            | Anthropic API or Ollama, behind one `LLMClient` interface       |
| Data           | PostgreSQL 16, Redis, ChromaDB or Qdrant                        |
| Jobs           | Celery worker and beat                                          |
| Frontend       | React 18, TypeScript, Vite, Tailwind, React Query, Zustand      |
| Testing        | Pytest, Vitest, Testing Library                                 |
| Quality and CI | Ruff, mypy, ESLint, GitHub Actions                              |
| Deployment     | Docker Compose, Nginx, Let's Encrypt, Prometheus, Grafana       |

---

## Project Structure

```
CampusMind/
├── backend/
│   ├── app/
│   │   ├── agents/          # LangGraph graph, router, planner, summarizer
│   │   ├── analytics/       # Admin metrics and question clustering
│   │   ├── api/             # REST routes and the notification WebSocket
│   │   ├── auth/            # Login, tokens, roles, rate limits
│   │   ├── conversations/   # History, feedback, retention
│   │   ├── ingestion/       # Extract, clean, chunk, tag, dedupe
│   │   ├── llm/             # LLMClient and its two providers
│   │   ├── models/          # SQLAlchemy models
│   │   ├── rag/             # Question answering and guardrails
│   │   ├── reminders/       # Date parsing, scheduling, delivery
│   │   └── vectorstore/     # Stores, embeddings, hybrid search
│   ├── alembic/             # Database migrations
│   ├── scripts/             # Seed data and a local PostgreSQL runner
│   └── tests/
├── frontend/
│   └── src/                 # Pages, components, API client, stores, tests
├── eval/                    # Test sets, experiments, charts, paper outline
├── infra/
│   ├── compose/             # Dev and production stacks
│   ├── docker/              # Backend and frontend images
│   ├── grafana/             # Dashboards
│   ├── nginx/               # Reverse proxy and HTTPS
│   ├── prometheus/          # Scrape config
│   └── scripts/             # Backups and first certificate
├── docs/                    # Architecture, schema, API, decisions, design notes
└── .github/workflows/       # CI
```

---

## Setup and Installation

### Prerequisites

- Docker
- Node.js 22
- [Ollama](https://ollama.com), or an Anthropic API key

### Quick Start

```bash
git clone https://github.com/Shashank17singh/CampusMind.git
cd CampusMind
cp .env.example .env
ollama pull llama3.2:1b
docker compose -f infra/compose/docker-compose.dev.yml up -d --build
docker compose -f infra/compose/docker-compose.dev.yml exec -e SEED_PASSWORD=choose-a-password backend python scripts/seed.py
cd frontend && npm install && npm run dev
```

Open <http://localhost:5173> and log in as `student@college.edu`, `faculty@college.edu` or `admin@college.edu` with that password. The seed creates 20 invented documents for a fictional college.

`llama3.2:1b` is small enough for a laptop, but its answers are weak, and a long summary is impractical. Use a hosted model or a larger local one for real work.

### Without Docker

Only PostgreSQL needs replacing, for the tests and migrations. Redis is still required to run the workers.

```bash
cd backend
uv sync
uv run python scripts/devdb.py alembic upgrade head
uv run pytest -m "not model"
cd ../frontend && npm install && npm test
```

---

## Running the Tests

```bash
cd backend
uv run ruff check . ../eval
uv run mypy app
uv run pytest -m "not model"   # Real PostgreSQL, Chroma and Qdrant, with fake LLM, OCR and Redis
uv run pytest -m model         # The real embedding model. Downloads about 130 MB

cd ../frontend
npm run lint
npm run typecheck
npm test                       # Chat, upload and the API client
```

CI runs these on every push: 1164 backend tests and 24 frontend tests pass on Linux.

---

## Evaluation

The [evaluation suite](eval/README.md) measures retrieval, faithfulness, refusals, latency and cost, and compares chunk sizes, retrieval strategies and models.

There are no results to report yet. It needs real college documents, so `eval/results/synthetic-smoke/` only shows that the tools run.

---

## Deployment

The [deployment guide](docs/deploy-vm.md) covers running everything on one cloud VM with the production compose file. It has not been tried on a real server yet.

CI builds the backend and frontend images and pushes them to GHCR on every push to `main`.

| Variable                                    | Required           | Purpose                                                                  |
| ------------------------------------------- | ------------------ | ------------------------------------------------------------------------ |
| `SECRET_KEY`                                | Yes in production  | Signs tokens, at least 32 characters                                     |
| `DATABASE_URL`, `REDIS_URL`                 | Yes in production  | Datastores. Local defaults apply otherwise                               |
| `LLM_PROVIDER`                              | No                 | `anthropic` or `ollama`. `ollama` by default                             |
| `LLM_MODEL`                                 | No                 | Overrides the provider's default model                                   |
| `ANTHROPIC_API_KEY`                         | For the hosted LLM | Anthropic API key                                                        |
| `LLM_MAX_CONCURRENCY`                       | No                 | Parallel summarizer calls. Use 1 with Ollama                             |
| `VECTOR_BACKEND`                            | No                 | `chroma` or `qdrant`. Use `qdrant` once an API and a worker run together |
| `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS` | No                 | Chunking                                                                 |
| `RERANK_ENABLED`                            | No                 | Adds the cross-encoder. Needs about 600 MB more memory                   |
| `ALLOWED_EMAIL_DOMAIN`                      | No                 | Who may register, comma-separated. `college.edu` by default              |
| `SMTP_HOST` and related                     | No                 | Mail. If empty, messages are logged and not sent                         |
| `COOKIE_SECURE`                             | Yes in production  | Must be `true` in production                                             |
| `CONVERSATION_RETENTION_DAYS`               | No                 | Older conversations are deleted. `0` keeps everything                    |

Every variable is listed in [.env.example](.env.example).

---

## Troubleshooting

| Symptom                                            | Cause and fix                                                                                     |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Nobody can log in after registering                | SMTP is not set, so no verification mail arrives. Set `SMTP_HOST`, or see the logs in development |
| A document stays at `uploaded`                     | The Celery worker is not running, or cannot reach Redis                                           |
| A document fails with "needs OCR"                  | Install Tesseract, or set `TESSERACT_CMD`                                                         |
| New documents do not appear in answers             | With Chroma's local mode, the API and worker do not share data. Use `VECTOR_BACKEND=qdrant`       |
| Every answer is a refusal                          | No document is `ready`, or `RAG_SIMILARITY_THRESHOLD` is too high for your embedding model        |
| Answers time out with Ollama                       | Raise `LLM_TIMEOUT_SECONDS` and set `LLM_MAX_CONCURRENCY=1`                                       |
| Signed out when two tabs are open                  | Two tabs refreshed at once, which looks like token theft. Log in again                            |
| Summaries say they were not fully checked          | Some LLM calls failed. Read the worker logs                                                       |
| A login limit does not apply per user behind Nginx | Confirm `X-Forwarded-For` reaches uvicorn with `--proxy-headers`                                  |

---

## Known Limitations

- No container has been started. CI builds both images, but the compose services, Nginx, certificates, backups and Grafana have not run.
- Redis, Celery workers and beat, and SMTP have not run for real. Tests use fakes and eager tasks.
- The hosted LLM client has only been tested against a stub, because it needs an API key.
- Tesseract OCR and the re-ranker model have not been run. OCR tests use a stub.
- Clickable citation chips are proven by component tests only, because `llama3.2:1b` never wrote a `[n]` marker in the browser runs.
- Long summaries on `llama3.2:1b` are not usable: 40 pages took 47.5 minutes and produced one bullet. See [the notes](docs/notes/summarizer.md).
- There are no evaluation numbers, because that needs real documents. The experiment grid has only run on a synthetic corpus, without the re-ranker.
- Chat has no per-user rate limit, only Nginx's limit per IP address, so one account could run up a hosted LLM bill.
- When Redis is down, each failed cache call waits about 4 seconds before carrying on.
- Google sign-in is not built.

---

## Deep Codebase Analysis

| File or Folder                       | Purpose / Details                                                     |
| ------------------------------------ | --------------------------------------------------------------------- |
| `backend/app/agents/graph.py`        | The LangGraph graph, with the safety wrapper around every node        |
| `backend/app/agents/router.py`       | Keyword rules first, then the LLM, for sorting a message into intents |
| `backend/app/agents/scheduler.py`    | The study plan, computed in code so its rules always hold             |
| `backend/app/agents/summarizer.py`   | Map-reduce summaries, with formula checks and an LLM verify pass      |
| `backend/app/rag/pipeline.py`        | Rewrite, retrieve, refuse or answer, and stream                       |
| `backend/app/rag/guardrails.py`      | Injection handling and streaming redaction                            |
| `backend/app/vectorstore/hybrid.py`  | Reciprocal rank fusion over vector and BM25 results                   |
| `backend/app/ingestion/pipeline.py`  | The steps a document passes through, with status updates              |
| `backend/app/reminders/scheduler.py` | Due reminders, idempotency keys and `FOR UPDATE SKIP LOCKED`          |
| `backend/app/auth/service.py`        | Registration, login, refresh rotation and reuse detection             |
| `frontend/src/api/client.ts`         | API client with token refresh                                         |
| `frontend/src/lib/sse.ts`            | Reads the streamed answer                                             |
| `eval/experiments.py`                | The grid of chunk sizes, retrieval strategies and models              |

### Documentation

| Topic                  | Document                                  |
| ---------------------- | ----------------------------------------- |
| Architecture           | [architecture.md](docs/architecture.md)   |
| Database schema        | [schema.md](docs/schema.md)               |
| REST API               | [api.md](docs/api.md)                     |
| Decisions              | [decisions.md](docs/decisions.md)         |
| Design notes by module | [docs/notes/](docs/notes/)                |
| Deployment             | [deploy-vm.md](docs/deploy-vm.md)         |
| Evaluation             | [eval/README.md](eval/README.md)          |
| Paper outline          | [paper_outline.md](eval/paper_outline.md) |

---

## License

MIT - see [LICENSE](LICENSE).
