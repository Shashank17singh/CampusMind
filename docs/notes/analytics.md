# Admin analytics

Code: `backend/app/analytics/`, `backend/app/api/analytics.py`. Document management is in `backend/app/api/documents.py`.

## How to run

All reports are under `/api/v1/admin/analytics/` and are admin-only. They take `?from=YYYY-MM-DD&to=YYYY-MM-DD` (default: the last 30 days) and `?format=csv`.

## How to test

```bash
pytest tests/test_analytics.py
```

## The queries and their indexes

| Report | Query shape | Index used |
|---|---|---|
| Daily activity | `WHERE role = 'user' AND created_at` in range, `GROUP BY day` with `count(DISTINCT user_id)` | `ix_messages_role_created (role, created_at)` |
| Average latency, intents, cost | `WHERE role = 'assistant' AND created_at` in range, then aggregate | `ix_messages_role_created` |
| Feedback ratio | `WHERE created_at` in range, `count(*) FILTER (WHERE rating = 1)` | `ix_feedback_created_at` |
| Top documents | `jsonb_array_elements(citations)` joined to `documents`, `GROUP BY` id | `ix_messages_role_created`, then the `documents` primary key |
| Unanswered | `WHERE answered IS FALSE AND created_at` in range | Partial index `ix_messages_unanswered` |
| Question for an answer | Correlated `ORDER BY created_at DESC LIMIT 1` in the same conversation | `ix_messages_conversation_role_created` |
| Knowledge gaps | Unanswered or joined to a `rating = -1` feedback row | `ix_messages_role_created`, `uq_feedback_message_user` |

Example, daily activity:

```sql
SELECT date(timezone(interval '05:30', created_at)) AS day,
       count(DISTINCT user_id) AS active_users,
       count(*) AS questions
FROM messages
WHERE role = 'user'
  AND created_at >= '2026-05-31T18:30:00Z'
  AND created_at <  '2026-06-30T18:30:00Z'
GROUP BY day
ORDER BY day;
```

The filter compares the raw column to a range, so an index applies. Wrapping `created_at` in a function inside `WHERE` would rule the index out. The range is computed in Python, and `day` is only computed in the `SELECT`. `TestIndexes` confirms the plan on real PostgreSQL.

## Key decisions

**1. Aggregate in SQL, and cluster in Python only where needed.** Counting in the database means reports stay fast as data grows. Clustering needs embeddings, so that part runs in Python, capped at the latest 2,000 questions.

**2. Greedy clustering around a fixed leader.** Questions are processed most-common first, and each joins the first group whose leader is over 0.85 similar. The leader is the commonest wording, so it reads well in the table, and because it never moves a group cannot drift. *Rejected:* k-means, which needs the number of groups in advance, and which has no notion of "these are not similar to anything".

**3. Days are in college time.** A question at 01:00 in India belongs to that morning, not the previous UTC day. The offset is a constant, because not every PostgreSQL build ships timezone data, and India has no daylight saving.

## More design points

- **Knowledge gaps.** Unanswered and thumbs-down questions are grouped at a looser 0.75, because the goal is a topic list, not duplicates. This tells faculty what to upload.
- **CSV injection.** A cell starting with `=` can run as a formula in Excel, and questions are user text, so such cells are prefixed with an apostrophe.
- **Cost.** Prices live in a table in `service.py` and will go out of date. Local models count as free.
- **Delete removes vectors.** The document endpoint deletes vectors, the file and the row, and clears the answer cache.

## Known limits

- Cost covers tokens recorded on messages, so background work such as summaries and auto-titles is not included.
- The report's UTC offset is fixed to India.
- Clustering loads the embedding model in the API process.
- Past the 2,000 most recent questions, older ones are not clustered.
