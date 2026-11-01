# Document ingestion

Code: `backend/app/ingestion/`.

## How to run

```bash
cd backend
uv sync
python scripts/devdb.py alembic upgrade head    # local Postgres, no Docker needed
celery -A app.celery_app worker --loglevel=info # needs Redis
uvicorn app.main:app --reload
```

## How to test

```bash
cd backend
pytest                                  # real Postgres, fake OCR, fake LLM
python tests/fixtures/make_fixtures.py  # rebuild the sample files
```

The test `test_real_tesseract_reads_the_scanned_pdf` is skipped until Tesseract is installed.

## Why 800 tokens, and how to justify it

A chunk has to do two jobs that pull in opposite directions. **Retrieval** wants small chunks, because one topic gives a sharper embedding. **Answering** wants large chunks, because a rule and its exceptions must arrive together. College regulations run to several paragraphs per clause, so 800 tokens (about 600 words, or a page) keeps a whole clause. The 120-token overlap (15%) means a sentence near a boundary appears whole in at least one chunk.

800 is a starting assumption, not a finding. There is also a known catch: **bge-small-en-v1.5 reads only 512 tokens**. The vector store embeds longer chunks in overlapping windows and averages the vectors, so no text is ignored, but an average of several topics can be a blurrier vector than one 400-token chunk. Expect the experiment below to take this seriously.

### The experiment

The evaluation suite runs this. For sizes 400, 800 and 1200 (overlap held at 15%):

1. Ingest the same real documents at each size.
2. Run the same 150 labelled questions.
3. Record precision@5, recall@5 and MRR, then answer faithfulness and cost per query.
4. Keep the size with the best recall@5 at an acceptable context size. Recall comes first because the LLM cannot cite a passage that was never retrieved.

Report the numbers per question category. Tables and multi-hop questions can favour a different size than simple facts, and that is a finding worth writing up. Pair the results with a significance test such as a paired bootstrap on per-question scores, because 150 questions can easily produce a gap that is just noise.

## Key decisions

**1. Rules first, LLM only when needed, for tagging.** Most documents announce their type in the filename or first lines, so rules are instant, free and predictable. The LLM runs only when `doc_type` is unknown. Its reply is validated by Pydantic, so a bad or manipulated reply can only produce an allowed value, and an LLM outage does not stop ingestion. *Rejected:* LLM for everything, which is slower, costs money per upload, and sends every document to a model for no gain.

**2. Text cleaning uses NFC plus a ligature table, not NFKC.** NFKC fixes ligatures but also turns x² into x2, silently changing formulas in engineering notes. *Rejected:* NFKC. There is a test for this.

**3. Word shingles with Jaccard similarity for near-duplicates.** A new circular version usually changes a date or a paragraph, which leaves most 5-word sequences intact, so similarity stays high (the threshold is 0.7). It is exact, needs no model, and is easy to explain. Comparison is limited to the 200 most recent circulars in the department. The old version is superseded only after the new one is `ready`, so a failed upload can never hide a valid circular. *Rejected:* embedding similarity, which scores two different exam circulars as close because they share a topic.

## Known limits

- Heading detection uses font size and weight, so a PDF with flat styling becomes one section.
- Text is read top to bottom, so two-column PDFs can interleave.
- Tables on scanned pages come through as OCR text, not Markdown.
- "well-\nknown" becomes "wellknown". Lowercase hyphen joins cannot tell these apart.
- A line in the body that repeats on over 60% of pages is removed like a header.
