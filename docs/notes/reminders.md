# Reminders and deadlines

Code: `backend/app/reminders/`, `backend/app/api/reminders.py`, `backend/app/api/ws.py`, `backend/app/email.py`.

## How to run

```bash
celery -A app.celery_app worker --loglevel=info
celery -A app.celery_app beat --loglevel=info   # exactly one beat process
```

Leave `SMTP_HOST` empty in development, and emails are logged instead of sent.

## How to test

```bash
pytest tests/test_reminders.py
```

Parsing tests use the real `dateparser` with a fixed "now". Delivery tests use real Postgres, including a real row lock for the two-worker case. Redis and SMTP are stubbed, so delivery through a real Redis, a real mail server and real Celery beat has not been run.

## Key decisions

**1. The LLM never computes a date.** It only copies the date words out of the message. `resolve_due` turns them into a datetime, so an LLM cannot invent a deadline, and a wrong answer would at worst be the wrong words, which then fail validation. This also keeps working when the LLM is down, because rules can pick out the same words.

**2. dateparser needed a layer in front of it.** Testing showed it returns nothing for "next monday", "this friday", "tonight" and "friday evening". It reads "by friday" as midnight at the start of Friday, a full day too early, and silently chooses a meaning for "03/04". So phrases are normalised first, a date with no time means 23:59 on that day, and day/month ambiguity becomes a question. Writing weekdays as ISO dates was tried first, and day-first parsing read `2026-11-09` as 11 September, so the month is written out instead.

**3. "Exactly once" means one row and one claimant, and the limit is stated.** A unique idempotency key means a repeat tick cannot create a second row. `FOR UPDATE SKIP LOCKED` means two workers cannot claim one row. Because in-app notifications are rows in the database, they really are exactly once. Email is different: if a crash lands after the mail server accepts a message and before the commit, the next tick sends it again. No design can make an external system part of a database transaction, so email is **at-least-once**. Sending a reminder twice is safer than missing it, so that is the side of the trade-off chosen. *Rejected:* claiming "exactly once" without qualification.

## Edge cases

| Case | Behaviour |
|---|---|
| Past date | Refused, with a question in chat and a 422 from the API |
| Recurring | Each occurrence's key contains its due date, so alerts repeat without colliding |
| Recurring, after downtime | Skips forward to the next future occurrence |
| Timezone change | `due_at` is stored as an instant, so the deadline does not move. Messages show the user's current timezone |
| Deleted assignment | Reminders become `cancelled`, and unsent alerts are excluded when delivered |
| Failed email | Retried next tick. One bad address does not block anyone else |
| Reminder set after a lead time has passed | That alert is skipped, not sent immediately |

## Known limits

- A student who joins a class after an assignment is posted gets no reminder for it.
- Beat must run as a single process, or ticks double. Idempotency keeps that harmless, but it wastes work.
- The rule-based title extractor is simple. It works for common phrasing, and the LLM handles the rest.
