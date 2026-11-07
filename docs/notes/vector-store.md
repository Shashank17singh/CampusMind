# Vector store and retrieval

Code: `backend/app/vectorstore/`, `eval/retrieval_bench.py`.

## How to run

Set `VECTOR_BACKEND=chroma` or `qdrant`. The embedding model downloads on first use (about 130 MB). The re-ranker is off by default, because it needs roughly another 600 MB of memory. Turn it on with `RERANK_ENABLED=true`.

```bash
cd backend
python ../eval/retrieval_bench.py ../eval/data/questions.json --output ../eval/results
```

## How to test

```bash
pytest -m "not model"   # both real stores, real BM25, hashed embeddings
pytest -m model         # the real bge-small model
```

Every store test runs against both Chroma and Qdrant, so the two cannot quietly drift apart. No unit test runs the real cross-encoder, because it is a 1.1 GB download. The experiment grid has run it on the synthetic corpus.

## Key decisions

**1. Reciprocal Rank Fusion rather than adding scores.** Cosine similarity runs from -1 to 1, and BM25 is unbounded and shifts with corpus size. Adding them needs a normalisation that breaks when the corpus changes. RRF uses only ranks: each item scores `weight / (60 + rank)` per list. An item near the top of both lists beats one at the very top of a single list. The constant 60 comes from the original RRF paper, and it stops the first few ranks from dominating. *Rejected:* weighted score sum.

**2. Long chunks are embedded as averaged windows.** bge-small reads 512 tokens and silently ignores the rest, so an 800-token chunk would lose its last 288 tokens from the vector. Chunks past the limit are split into overlapping windows, embedded separately, then averaged and normalised. `test_real_model_notices_text_past_its_512_token_limit` proves this on the real model: a query about text beyond token 512 scores higher than with plain truncation. *Rejected:* letting the model truncate, and shrinking chunks silently, which would change the default without evidence.

**3. A missing department means "everyone", not "no match".** A circular for the whole college has no department. A plain equality filter would hide it from every student who has one, which is the most common filter in the app. So filters match the value or empty. Both stores and BM25 share this meaning, and there are tests for each.

## More design points

- **Why BM25 at all?** Embeddings blur exact identifiers. "CS301" or "Clause 7.2" are found reliably by keyword search.
- **The query prefix.** BGE models were trained with a retrieval instruction on queries only, and leaving it off lowers quality.
- **Title and heading in the embedded text.** A chunk reading "This fee is non-refundable" means nothing alone. Prefixing the title and section gives the vector its context. The stored text stays clean for citations.
- **Why the BM25 index rebuilds.** `rank_bm25` has no incremental updates, so it rebuilds when a corpus fingerprint changes. This costs seconds at college scale and would need replacing at millions of chunks.
- **Why `similarities` travels with results.** RRF scores only compare chunks within a query. The RAG layer's "is this good enough to answer from?" check needs a real similarity.

## Known limits

- With Chroma's persistent client, two processes must not share a folder. Set `CHROMA_HOST` to use a Chroma server once an API and a worker run side by side. Qdrant has no such limit.
- Each process keeps its own BM25 copy in memory, so memory use grows with the number of workers.
