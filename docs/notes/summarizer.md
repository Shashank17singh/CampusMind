# Summarizer

Code: `backend/app/agents/summarizer.py`, `backend/app/agents/summary_jobs.py`, `backend/app/api/summaries.py`.

## How to run

`POST /api/v1/summaries` with `{document_id, mode}`, or `POST /api/v1/summaries/upload` with a file. Both return `202` with an id to poll. Modes are `brief`, `detailed`, `exam_notes` and `flashcards`.

## How to test

```bash
pytest tests/test_summarizer.py
```

The 40-page test uses a scripted LLM to prove batching, the five-call concurrency limit and token counting. For a real measurement, see below.

## Measured results

One real run, with a 40-page synthetic PDF (40 chunks, 16,620 tokens), `brief` mode, on a laptop CPU with 8 GB of memory, using Ollama and `llama3.2:1b`, one call at a time:

| Measure | Result |
|---|---|
| Time | 47.5 minutes |
| Successful LLM calls | 10 |
| Tokens in, out | 19,599 and 8,524 |
| Items produced | 1 |
| Verified | No |

An earlier run with five parallel calls took 61 minutes while other work was using the machine, and it lost more batches. That is why `LLM_MAX_CONCURRENCY` exists.

**This setup does not produce a usable summary of a long document.** The pipeline completed and correctly marked the result unverified, but the output was one bullet. The small model is the limit here. A hosted model, or a larger local one on better hardware, has not been measured. Do that before quoting a figure for this feature.

## Key decisions

**1. Map-reduce, because a document will not fit in one prompt.** Chunks are condensed in batches, then combined. Each batch carries page labels, so page references survive into the final summary. If the condensed points are still too long, they are condensed again in rounds.

**2. Two independent checks, one of them without an LLM.** An LLM is not a reliable judge of formulas, so a formula is kept only if the same string, ignoring spacing, exists in the source. That is a plain comparison and cannot hallucinate. Then a second LLM pass reads each item beside its source and drops unsupported items. The drop count is reported, so readers can see how much was removed. If the checker itself fails, items are kept and the summary is marked `verified: false`, because silently claiming it was checked would be worse.

**3. A student's upload never enters the document store.** Search results are treated as official. If an uploaded file were indexed, a student's own notes could be cited as college policy. Uploads are read, summarized and deleted, even if the job fails.

## More design points

- **Why max five concurrent calls?** It bounds load on a hosted API. It is configurable through `LLM_MAX_CONCURRENCY`, because Ollama handles one request at a time by default, and sending it five makes requests time out in its queue.
- **Why one result shape for every mode?** Groups of items let verification, export and rendering work the same way for all four modes.
- **Why background jobs?** A long document takes longer than a chat turn should. Chat summarizes short documents inline, and queues long ones.
- **Why section headings go to the planner.** In a multi-step request, the planner needs topics. Headings are free, while a summary might still be running.

## Known limits

- Verification is only as good as the model doing it.
- Bare formulas with no LaTeX delimiters are not checked by the string comparison.
- PDF export shows LaTeX as source text, not rendered equations.
- If batches are lost, the summary is marked unverified, but it still covers less of the document than it appears to.
