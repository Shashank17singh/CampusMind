# PostgreSQL schema

Target: PostgreSQL 16. The source of truth is `backend/app/models/` and the Alembic migrations. This page explains the design.

All primary keys are `uuid` except on the two append-only tables, `audit_logs` and `agent_traces`, which use `bigint identity` because they are written often and never referenced from elsewhere. All timestamps are `timestamptz` in UTC.

The first nine tables and `audit_logs` are the core schema. The tables under [Added for later features](#added-for-later-features) serve the agents, planner, reminders, authentication and summarizer.

## Enums

Stored as `text` with a `CHECK` constraint rather than native Postgres enums. Adding a value is then a one-line migration and does not need `ALTER TYPE`.

| Name | Values |
|---|---|
| `user_role` | `student`, `faculty`, `admin` |
| `doc_type` | `syllabus`, `circular`, `notice`, `regulation`, `notes`, `contact_list` |
| `doc_status` | `uploaded`, `extracting`, `chunking`, `embedding`, `ready`, `failed` |
| `message_role` | `user`, `assistant`, `system` |
| `feedback_reason` | `wrong`, `outdated`, `incomplete`, `not_relevant` |
| `reminder_status` | `pending`, `done`, `cancelled` |
| `plan_item_type` | `learn`, `revise`, `practice` |
| `plan_item_status` | `pending`, `done`, `missed` |

## Tables

### users

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `email` | `text` | not null |
| `password_hash` | `text` | Argon2, null for OAuth-only accounts |
| `full_name` | `text` | not null |
| `roll_number` | `text` | nullable |
| `role` | `text` | `user_role`, default `student` |
| `department` | `text` | nullable |
| `semester` | `smallint` | nullable, 1 to 8 |
| `section` | `text` | nullable |
| `timezone` | `text` | default `Asia/Kolkata` |
| `email_verified_at` | `timestamptz` | nullable |
| `is_active` | `boolean` | default true |
| `failed_login_count` | `integer` | default 0 |
| `locked_until` | `timestamptz` | nullable |
| `created_at`, `updated_at` | `timestamptz` | |

Indexes: unique `lower(email)`; unique `roll_number` where not null; `(department, semester)` for class-wide reminders.

### documents

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `title` | `text` | not null |
| `filename` | `text` | original name |
| `mime_type` | `text` | not null |
| `size_bytes` | `bigint` | not null |
| `sha256` | `char(64)` | not null |
| `storage_path` | `text` | not null |
| `doc_type` | `text` | `doc_type`, nullable until tagged |
| `department` | `text` | nullable, null means college-wide |
| `semester` | `smallint` | nullable |
| `academic_year` | `text` | nullable, for example `2026-27` |
| `status` | `text` | `doc_status`, default `uploaded` |
| `progress` | `smallint` | 0 to 100 |
| `error_message` | `text` | nullable |
| `page_count` | `integer` | nullable |
| `syllabus_outline` | `jsonb` | nullable, cached units and topics |
| `uploader_id` | `uuid` | FK `users`, `ON DELETE SET NULL` |
| `superseded_by_id` | `uuid` | FK `documents`, `ON DELETE SET NULL` |
| `uploaded_at` | `timestamptz` | not null |
| `updated_at` | `timestamptz` | not null |

Indexes: unique `sha256`; `(status)`; `(doc_type, department, semester)`; `(uploader_id)`; partial `(uploaded_at desc)` where `superseded_by_id is null`.

### chunks_meta

The `id` is also the point ID in the vector store, so a search hit joins straight back to its metadata. Chunk text is kept here because BM25 and citations both need it and the vector store is meant to be replaceable.

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `document_id` | `uuid` | FK `documents`, `ON DELETE CASCADE` |
| `chunk_index` | `integer` | order within the document |
| `text` | `text` | not null |
| `token_count` | `integer` | not null |
| `page_number` | `integer` | page or slide, nullable |
| `section_heading` | `text` | nullable |
| `is_table` | `boolean` | default false |
| `created_at` | `timestamptz` | |

Indexes: unique `(document_id, chunk_index)`.

### conversations

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `user_id` | `uuid` | FK `users`, `ON DELETE CASCADE` |
| `title` | `text` | nullable until auto-titled |
| `created_at` | `timestamptz` | |
| `last_message_at` | `timestamptz` | not null |

Indexes: `(user_id, last_message_at desc)`; `(last_message_at)` for retention.

### messages

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `conversation_id` | `uuid` | FK `conversations`, `ON DELETE CASCADE` |
| `user_id` | `uuid` | FK `users`, denormalised so search avoids a join |
| `role` | `text` | `message_role` |
| `content` | `text` | not null |
| `citations` | `jsonb` | default `[]` |
| `intent` | `text` | nullable |
| `confidence` | `text` | `high`, `medium`, `low`, nullable |
| `answered` | `boolean` | nullable, false when the answer was refused |
| `latency_ms` | `integer` | nullable |
| `tokens_in`, `tokens_out` | `integer` | nullable |
| `model_name` | `text` | nullable |
| `trace_id` | `uuid` | nullable |
| `search_vector` | `tsvector` | generated from `content` |
| `created_at` | `timestamptz` | |

Indexes: `(conversation_id, created_at)`; `(conversation_id, role, created_at)`; `(role, created_at)`; GIN `(search_vector)`; `(user_id, created_at desc)`; partial `(created_at)` where `answered = false`.

### feedback

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `message_id` | `uuid` | FK `messages`, `ON DELETE CASCADE` |
| `user_id` | `uuid` | FK `users`, `ON DELETE CASCADE` |
| `rating` | `smallint` | `CHECK (rating IN (-1, 1))` |
| `reason` | `text` | `feedback_reason`, nullable |
| `comment` | `text` | nullable |
| `created_at` | `timestamptz` | |

Indexes: unique `(message_id, user_id)`; `(created_at)`.

### tasks

Assignments posted by faculty for a class. Each student in the class gets a linked row in `reminders`.

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `created_by` | `uuid` | FK `users`, `ON DELETE SET NULL` |
| `title` | `text` | not null |
| `description` | `text` | nullable |
| `subject` | `text` | nullable |
| `department` | `text` | not null |
| `semester` | `smallint` | not null |
| `section` | `text` | nullable, null means all sections |
| `due_at` | `timestamptz` | not null |
| `deleted_at` | `timestamptz` | nullable, soft delete |
| `created_at` | `timestamptz` | |

Indexes: `(department, semester, due_at)`.

### reminders

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `user_id` | `uuid` | FK `users`, `ON DELETE CASCADE` |
| `task_id` | `uuid` | FK `tasks`, `ON DELETE CASCADE`, nullable |
| `title` | `text` | not null |
| `subject` | `text` | nullable |
| `due_at` | `timestamptz` | not null |
| `remind_before` | `integer[]` | minutes, default `{1440,120}` |
| `recurrence` | `text` | nullable, for example `weekly` |
| `status` | `text` | `reminder_status`, default `pending` |
| `snoozed_until` | `timestamptz` | nullable |
| `source_text` | `text` | original wording, nullable |
| `created_at`, `updated_at` | `timestamptz` | |

Indexes: `(user_id, status, due_at)`; unique `(task_id, user_id)` where `task_id` is not null.

### study_plans

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `user_id` | `uuid` | FK `users`, `ON DELETE CASCADE` |
| `title` | `text` | not null |
| `inputs` | `jsonb` | subjects, exam dates, hours, ratings |
| `start_date`, `end_date` | `date` | not null |
| `is_active` | `boolean` | default true |
| `created_at`, `updated_at` | `timestamptz` | |

Indexes: `(user_id, is_active)`.

### audit_logs

| Column | Type | Notes |
|---|---|---|
| `id` | `bigint identity` | PK |
| `actor_id` | `uuid` | FK `users`, `ON DELETE SET NULL` |
| `action` | `text` | for example `auth.login` |
| `entity_type`, `entity_id` | `text` | nullable |
| `ip_address` | `inet` | nullable |
| `user_agent` | `text` | nullable |
| `details` | `jsonb` | default `{}` |
| `created_at` | `timestamptz` | |

Indexes: `(created_at desc)`; `(actor_id, created_at desc)`; `(action, created_at desc)`.

## Added for later features

### plan_items

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `plan_id` | `uuid` | FK `study_plans`, `ON DELETE CASCADE` |
| `date` | `date` | not null |
| `start_time`, `end_time` | `time` | not null |
| `subject`, `topic` | `text` | not null |
| `type` | `text` | `plan_item_type` |
| `status` | `text` | `plan_item_status`, default `pending` |

Indexes: `(plan_id, date, start_time)`.

### reminder_notifications

One row per reminder, channel and fire time. The scheduler claims rows with `FOR UPDATE SKIP LOCKED`, and the unique key means a repeat beat tick cannot create a second row.

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `reminder_id` | `uuid` | FK `reminders`, `ON DELETE CASCADE` |
| `user_id` | `uuid` | FK `users`, `ON DELETE CASCADE` |
| `channel` | `text` | `in_app` or `email` |
| `fire_at` | `timestamptz` | not null |
| `idempotency_key` | `text` | not null |
| `sent_at` | `timestamptz` | nullable |
| `read_at` | `timestamptz` | nullable |
| `payload` | `jsonb` | default `{}` |

Indexes: unique `(idempotency_key)`; partial `(fire_at)` where `sent_at is null`; `(user_id, sent_at desc)`.

### agent_traces

| Column | Type | Notes |
|---|---|---|
| `id` | `bigint identity` | PK |
| `trace_id` | `uuid` | not null |
| `step` | `integer` | not null |
| `node` | `text` | not null |
| `input`, `output` | `jsonb` | not null |
| `latency_ms` | `integer` | not null |
| `tokens_in`, `tokens_out` | `integer` | default 0 |
| `error` | `text` | nullable |
| `created_at` | `timestamptz` | |

Indexes: `(trace_id, step)`; `(created_at)`.

### refresh_tokens

Only the hash of a token is stored. `family_id` ties rotated tokens together, so reuse of a spent token revokes the whole family.

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `user_id` | `uuid` | FK `users`, `ON DELETE CASCADE` |
| `family_id` | `uuid` | not null |
| `token_hash` | `char(64)` | not null |
| `expires_at` | `timestamptz` | not null |
| `used_at`, `revoked_at` | `timestamptz` | nullable |
| `created_at` | `timestamptz` | |

Indexes: unique `(token_hash)`; `(family_id)`; `(user_id)`.

### summaries

| Column | Type | Notes |
|---|---|---|
| `id` | `uuid` | PK |
| `user_id` | `uuid` | FK `users`, `ON DELETE CASCADE` |
| `document_id` | `uuid` | FK `documents`, `ON DELETE CASCADE`, null for an uploaded file |
| `source_name` | `text` | the document title or uploaded filename |
| `mode` | `text` | `brief`, `detailed`, `exam_notes`, `flashcards` |
| `status` | `text` | `pending`, `ready`, `failed` |
| `error_message` | `text` | nullable |
| `content` | `jsonb` | null until ready |
| `dropped_count` | `integer` | bullets removed by the verifier |
| `tokens_in`, `tokens_out` | `integer` | default 0 |
| `duration_ms` | `integer` | nullable |
| `created_at` | `timestamptz` | |

Indexes: `(user_id, created_at desc)`; `(document_id, mode)`.
