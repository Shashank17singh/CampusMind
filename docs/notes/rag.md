# RAG question answering

Code: `backend/app/rag/`, `backend/app/api/ask.py`.

## How to run

```bash
curl -N -X POST localhost:8000/api/v1/ask \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question": "What attendance do I need?"}'
```

## How to test

```bash
pytest tests/test_rag.py
```

These tests use real Chroma and BM25 with a scripted LLM. Prompt behaviour was also checked once on the real stack (bge-small, Chroma, BM25, Postgres, Ollama with `llama3.2:1b`): two factual questions were answered correctly, and an off-topic question was refused with no LLM call. The small model sometimes left out `[n]` markers. Those answers are marked low confidence, not given an invented citation.

## Key decisions

**1. Two layers against prompt injection.** A malicious PDF could say "ignore your instructions". Layer one removes sentences matching known attack patterns before the model sees them. Layer two wraps each source in `<source>` tags, strips any attempt to close the tag, and tells the model that sources are data. Neither is enough alone: patterns can be reworded, and a model can be persuaded. Together they make an attack harder, and neither claims to make it impossible. *Rejected:* trusting the system prompt alone.

**2. Any document change clears the answer cache, not just "related" ones.** Invalidating only on re-ingestion of a related document sounds enough, but the dangerous case is a new circular replacing an old rule: no cached answer can be linked to a document that did not exist when it was written, so a tracked approach would serve the old rule for up to an hour. A generation number in every key makes one `INCR` invalidate everything. Uploads are rare, so the extra cache misses cost little. *Rejected:* per-document tracking sets.

**3. Personal data is redacted during streaming, not after.** Once a token has reached the browser it cannot be recalled, so redaction must happen before sending. A number can span several tokens, so the stream holds back its last 64 characters until more text confirms that they are safe. Numbers survive only if they are printed in a `contact_list` source in that answer's context, so a contact list cannot unlock a number from some other document. *Rejected:* redacting the final text only, which leaks during streaming.

## More design points

- **Why two scores?** Retrieval ranks with RRF, but RRF cannot say whether results are any good. The "fewer than 3 above threshold" rule uses cosine similarity.
- **Refusing without the LLM.** With nothing relevant retrieved, the exact refusal is returned directly. It costs nothing and removes any chance of a made-up answer.
- **Why superseded circulars are excluded and dates are still shown.** Supersession only catches near-identical versions. A new fee circular worded differently is not detected, so both reach the model with dates, and the prompt says to trust the newest.
- **Why only the rewritten question goes to the answering call.** History is used only for rewriting. It keeps the prompt smaller and gives past messages no route into the final answer.
- **Threshold 0.45.** This is a starting value, not a measured one. The evaluation's unanswerable-detection accuracy is how to tune it.

## Known limits

- Injection patterns are English only and can be evaded by rewording.
- Phone patterns target Indian formats. Other formats pass through.
- If Redis is unavailable, caching is skipped, but a missed invalidation then leaves old entries live until they expire.
- Confidence is a heuristic from retrieval strength and citations, not a calibrated probability.
