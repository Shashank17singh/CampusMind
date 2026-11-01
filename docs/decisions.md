# Decisions and assumptions

Where a requirement was ambiguous, or two requirements pulled apart, the choice is recorded here so it can be revisited.

## Conflicts and gaps

| # | Issue | Decision |
|---|---|---|
| 1 | The core schema has ten tables, but the agents, planner, reminders, authentication and summarizer need more. | Added `agent_traces`, `plan_items`, `reminder_notifications`, `refresh_tokens` and `summaries`. |
| 2 | `doc_type` had five values, but answers keep phone numbers only for an "official contact list". | Added a sixth value, `contact_list`. Without it the rule could never apply. |
| 3 | The API is REST and WebSocket, but streamed answers need a one-way channel. | SSE for chat tokens, because they flow one way over a normal POST. WebSocket for notifications and ingestion status. |
| 4 | `tasks` needed a clear meaning. | It holds faculty-posted assignments. Each student gets a linked `reminders` row. |
| 5 | Chunks default to 800 tokens, but bge-small-en-v1.5 reads at most 512 tokens and silently drops the rest. | Keep 800. Chunks that are too long are embedded in overlapping windows whose vectors are averaged, so no text is lost. The evaluation tests whether a smaller size does better. |
| 6 | "Exactly once" reminder delivery cannot be guaranteed across SMTP. | The idempotency key guarantees one notification row and one claim. A crash between sending and recording could still lose or repeat an email, and this is stated openly in the design notes. |
| 7 | A hosted LLM provider had to be chosen. | `LLM_PROVIDER=anthropic` is the hosted option, with the model set by environment variable. |
| 8 | The evaluation needs 150 questions and the demo needs 20 documents, but no real college documents exist yet. | The tooling is built and tested on clearly labelled synthetic documents. Numbers from that corpus are not results for the paper. Everything must be rerun on real documents. |
| 9 | A persistent Chroma client suits one process, but the API and Celery worker are separate processes, which Chroma's local mode does not support. | Persistent client for single-process use. `CHROMA_HOST` switches to a Chroma server, and prod compose must use that or Qdrant. |
| 10 | "Filter by department" would hide college-wide documents, which have no department, from every student. | A filter matches the value or empty, in both stores and in BM25. |
| 11 | Cached answers could be cleared when a related document is re-ingested, but a new circular cannot be related to an answer written before it existed. | Any document change invalidates every cached answer, through a generation number in the key. |
| 12 | A student's file uploaded to the summarizer would otherwise enter the document store, where search treats everything as official. | Uploads are read, summarized and deleted. They never reach `documents` or the vector store. `summaries.document_id` is nullable for this. |
| 13 | Google OAuth is optional, and it needs real Google credentials to test. | Not built. Password login covers every flow, and a provider can be added later. |

## Local development without Docker

Tests run against real PostgreSQL 16 through the `pgserver` package, so migrations, `tsvector` search and `FOR UPDATE SKIP LOCKED` are tested for real on any machine. Redis is replaced by `fakeredis` in tests and Celery runs tasks eagerly.
