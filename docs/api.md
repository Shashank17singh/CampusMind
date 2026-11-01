# API contract

Base path: `/api/v1`. Requests and responses are JSON unless noted. Authenticated calls send `Authorization: Bearer <access token>`.

**Roles.** `public` needs no login, `any` is any signed-in user, `owner` is the signed-in user acting on their own data. `faculty*` means faculty limited to documents they uploaded or their own department's documents.

**Errors.** Every error has the shape `{"error": {"code": "...", "message": "...", "trace_id": "..."}}`.

**Pagination.** List endpoints take `?limit=&offset=` and return `{"items": [...], "total": n}`.

## Auth and profile

| Method | Path | Role | Request | Response |
|---|---|---|---|---|
| POST | `/auth/register` | public | `{email, password, full_name, roll_number?, department?, semester?, section?}` | `201 {id, email}` |
| POST | `/auth/verify-email` | public | `{token}` | `{verified: true}` |
| POST | `/auth/resend-verification` | public | `{email}` | `202` |
| POST | `/auth/login` | public | `{email, password}` | `{access_token, expires_in, user}`, sets refresh cookie |
| POST | `/auth/refresh` | public, cookie | none | `{access_token, expires_in}`, rotates cookie |
| POST | `/auth/logout` | any | none | `204`, revokes token family |
| POST | `/auth/forgot-password` | public | `{email}` | `202` |
| POST | `/auth/reset-password` | public | `{token, new_password}` | `204` |
| GET | `/auth/google/login` | public | none | `302` to Google |
| GET | `/auth/google/callback` | public | `?code&state` | `302` to app |
| GET | `/users/me` | any | none | `User` |
| PATCH | `/users/me` | any | `{full_name?, department?, semester?, section?, timezone?}` | `User` |

## Users (admin)

| Method | Path | Role | Request | Response |
|---|---|---|---|---|
| GET | `/admin/users` | admin | `?role&department&q` | page of `User` |
| PATCH | `/admin/users/{id}` | admin | `{role?, is_active?}` | `User` |
| POST | `/admin/users/import` | admin | multipart CSV | `{created, skipped, errors[]}` |

## Documents

| Method | Path | Role | Request | Response |
|---|---|---|---|---|
| POST | `/documents` | faculty, admin | multipart `file`, `title?`, `doc_type?`, `department?`, `semester?` | `202 Document`, or `200` with `{duplicate_of}` |
| GET | `/documents` | any | `?status&doc_type&department&semester&mine` | page of `Document` |
| GET | `/documents/{id}` | any | none | `Document` with `status`, `progress`, `error_message` |
| GET | `/documents/{id}/status` | any | none | `{status, progress, error_message}` |
| PATCH | `/documents/{id}` | faculty*, admin | `{title?, doc_type?, department?, semester?}` | `Document` |
| DELETE | `/documents/{id}` | faculty*, admin | none | `204`, also removes vectors |
| POST | `/documents/{id}/reingest` | faculty*, admin | none | `202 Document` |
| POST | `/documents/{id}/supersede` | faculty*, admin | `{superseded_by_id}` | `Document` |

## Chat and conversations

| Method | Path | Role | Request | Response |
|---|---|---|---|---|
| POST | `/conversations` | any | `{title?}` | `201 Conversation` |
| GET | `/conversations` | owner | pagination | page of `Conversation`, newest first |
| GET | `/conversations/search` | owner | `?q` | page of `{message_id, conversation_id, snippet, created_at}` |
| GET | `/conversations/export` | owner | none | JSON download of all history |
| GET | `/conversations/{id}/messages` | owner | pagination | page of `Message` |
| PATCH | `/conversations/{id}` | owner | `{title}` | `Conversation` |
| DELETE | `/conversations/{id}` | owner | none | `204` |
| POST | `/conversations/{id}/messages` | owner | `{content}` | SSE stream, below |
| POST | `/messages/{id}/feedback` | owner | `{rating, reason?, comment?}` | `Feedback` |

`POST /ask` (role `any`, body `{question, history?}`) is the same pipeline with no stored conversation. It answers directly from documents and sends the events below, without `meta`.

SSE events from `POST /conversations/{id}/messages`, sent as `event: <name>` with a JSON `data` line:

| Event | Data |
|---|---|
| `meta` | `{trace_id, intent, message_id}` |
| `token` | `{text}` |
| `citations` | `{citations: [{document_id, title, page, snippet}], confidence, suggested_followups}` |
| `done` | `{latency_ms, tokens_in, tokens_out}` |
| `error` | `{code, message, trace_id}` |

## Study plans

| Method | Path | Role | Request | Response |
|---|---|---|---|---|
| POST | `/study-plans` | any | `{exams: [{subject, date}], hours_per_weekday, completed_topics, difficulty, start_date?}` | `201 StudyPlan` with items |
| GET | `/study-plans` | owner | none | page of `StudyPlan` |
| GET | `/study-plans/{id}` | owner | none | `StudyPlan` with items |
| PATCH | `/study-plans/{id}/items/{item_id}` | owner | `{status}` | `PlanItem` |
| POST | `/study-plans/{id}/replan` | owner | `{from_date?}` | `StudyPlan` |
| GET | `/study-plans/{id}/export.ics` | owner | none | `text/calendar` |
| DELETE | `/study-plans/{id}` | owner | none | `204` |

## Reminders and assignments

| Method | Path | Role | Request | Response |
|---|---|---|---|---|
| POST | `/reminders` | any | `{title, due_at, subject?, remind_before?, recurrence?}` | `201 Reminder` |
| POST | `/reminders/parse` | any | `{text}` | `201 Reminder`, or `200 {clarification}` |
| GET | `/reminders` | owner | `?view=upcoming\|overdue\|done` | page of `Reminder` |
| POST | `/reminders/{id}/done` | owner | none | `Reminder` |
| POST | `/reminders/{id}/snooze` | owner | `{minutes}` | `Reminder` |
| DELETE | `/reminders/{id}` | owner | none | `204` |
| POST | `/assignments` | faculty, admin | `{title, description?, subject?, department, semester, section?, due_at}` | `201 Task` with `{reminders_created}` |
| GET | `/assignments` | faculty, admin | `?mine` | page of `Task` |
| DELETE | `/assignments/{id}` | faculty*, admin | none | `204`, cancels linked reminders |
| GET | `/notifications` | owner | `?unread` | page of `Notification` |
| POST | `/notifications/{id}/read` | owner | none | `204` |

## Summaries

| Method | Path | Role | Request | Response |
|---|---|---|---|---|
| POST | `/summaries` | any | `{document_id, mode}` | `202 {summary_id}` |
| POST | `/summaries/upload` | any | multipart `file`, `mode` | `202 {summary_id}` |
| GET | `/summaries` | owner | none | page of `Summary` |
| GET | `/summaries/{id}` | owner | none | `Summary` |
| GET | `/summaries/{id}/export` | owner | `?format=md\|pdf` | file download |

## Admin analytics and traces

All take `?from=&to=` dates. Each report also takes `?format=csv`.

| Method | Path | Role | Request | Response |
|---|---|---|---|---|
| GET | `/admin/analytics/overview` | admin | range | `{dau[], questions_per_day[], avg_latency_ms, feedback_ratio}` |
| GET | `/admin/analytics/top-questions` | admin | range | `[{question, count, examples[]}]` |
| GET | `/admin/analytics/unanswered` | admin | range | page of `{question, asked_at, user_id}` |
| GET | `/admin/analytics/intents` | admin | range | `[{intent, count}]` |
| GET | `/admin/analytics/top-documents` | admin | range | `[{document_id, title, citations}]` |
| GET | `/admin/analytics/cost` | admin | range | `[{day, tokens_in, tokens_out, cost_usd}]` |
| GET | `/admin/analytics/knowledge-gaps` | admin | range | `[{topic, count, examples[]}]` |
| GET | `/admin/feedback` | admin | `?rating&reason` | page of `Feedback` |
| GET | `/admin/traces/{trace_id}` | admin | none | `{trace_id, steps: [{node, input, output, latency_ms, tokens_in, tokens_out, error}]}` |

## Operations

| Method | Path | Role | Request | Response |
|---|---|---|---|---|
| GET | `/healthz` | public | none | `{status}`, liveness |
| GET | `/readyz` | public | none | `{status, checks}`, checks Postgres and Redis |
| GET | `/metrics` | internal | none | Prometheus text, blocked at Nginx |

`/healthz`, `/readyz` and `/metrics` are served at the root, outside `/api/v1`.

## WebSocket

`WS /api/v1/ws/notifications`, role `any`.

Browsers cannot set an `Authorization` header on a WebSocket, and a token in the URL would end up in access logs. So the client authenticates in its first message, and the server closes with code `4401` if that message is missing, invalid, or does not arrive within 5 seconds.

| Direction | Message |
|---|---|
| Client to server | `{"type": "auth", "token": "<access token>"}` |
| Server to client | `{"type": "ready"}` |
| Server to client | `{"type": "notification", "id", "title", "body", "due_at", "reminder_id"}` |
| Either | `{"type": "ping"}` and `{"type": "pong"}` |

Ingestion progress is read by polling `GET /documents/{id}/status`, not over this socket.
