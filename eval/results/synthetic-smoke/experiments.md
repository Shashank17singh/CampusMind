# Retrieval experiments

Questions: 29.

**Synthetic corpus and unverified questions. This shows the tooling works. These numbers are not results and must not be reported.**

## Chunk size (tokens)

| Setting | Precision@5 | Recall@5 | MRR | MRR change, 95% interval |
|---|---|---|---|---|
| 400 | 0.220 | 1.000 | 0.948 | +0.000 [+0.000, +0.000] (no clear difference) |
| 800 | 0.220 | 1.000 | 0.948 | baseline |
| 1200 | 0.220 | 1.000 | 0.948 | +0.000 [+0.000, +0.000] (no clear difference) |

## Embedding model

| Setting | Precision@5 | Recall@5 | MRR | MRR change, 95% interval |
|---|---|---|---|---|
| bge-small | 0.220 | 1.000 | 0.948 | baseline |
| all-MiniLM-L6-v2 | 0.222 | 1.000 | 0.979 | +0.031 [+0.000, +0.083] (no clear difference) |

## Retrieval strategy

| Setting | Precision@5 | Recall@5 | MRR | MRR change, 95% interval |
|---|---|---|---|---|
| vector | 0.226 | 1.000 | 0.958 | +0.010 [+0.000, +0.031] (no clear difference) |
| hybrid | 0.220 | 1.000 | 0.948 | baseline |
| hybrid+rerank | 0.216 | 1.000 | 0.972 | +0.024 [+0.000, +0.069] (no clear difference) |
