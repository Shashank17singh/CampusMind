# Chat history and feedback

Code: `backend/app/conversations/`, `backend/app/api/conversations.py`, `backend/app/agents/runner.py`.

## How to test

```bash
pytest tests/test_conversations.py
```

Turns run through the real graph with a scripted router and agent, so storing, streaming and tracing are tested together. Search uses real PostgreSQL full-text search.

## Key decisions

**1. No database connection is held during an LLM call.** A chat turn can take a minute, and a connection held that long would exhaust the pool under load. The turn commits the user's message, closes the session, streams, then opens a new short session to save the answer.

**2. Stop keeps what was generated.** The stop button closes the connection, which closes the server's generator partway through. A `finally` block stores the text produced so far with the intent `stopped`, so the conversation matches what the user saw. The agent's thread keeps running until its current LLM call ends, because Python cannot kill a thread.

**3. Message search uses PostgreSQL, not the vector store.** Users searching their history want the words they remember, and it has to be private to them. A generated `tsvector` column with a GIN index gives stemming and ranking without another service. `websearch_to_tsquery` accepts anything typed into a search box without a syntax error, and `user_id` is on `messages` so the filter needs no join.

## More design points

- **Why `user_id` on `messages`?** It is denormalised on purpose, so "only my messages" is a direct filter that cannot be forgotten in a join.
- **Auto-titling runs in the background.** It should not delay the first answer. If the user has already renamed the conversation, the task leaves it alone, and a failed LLM call falls back to the question's first six words.
- **Feedback upsert.** Changing your mind replaces the rating, so one user cannot stack votes on one message.
- **Retention.** A daily task deletes conversations idle for `CONVERSATION_RETENTION_DAYS`, and messages and feedback go with them through foreign key cascades. Zero keeps everything.

## Known limits

- Deleting a conversation deletes its feedback, so retention also shrinks the feedback history.
- A stopped answer has no citations, because they are produced at the end.
- Message ordering relies on timestamps, so two messages in the same instant could swap.
