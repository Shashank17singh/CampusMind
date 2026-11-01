# Research paper outline

Target: an IEEE conference paper, about 6 pages. Everything in brackets is yours to fill from your own experiments. Do not write a number here until it comes from a run on real documents with a verified test set.

## Title

CampusMind: Multi-Agent Retrieval-Augmented Generation for Answering Questions from Institutional Documents

## Abstract (150 to 200 words)

- Problem: students cannot easily find answers across scattered, changing official documents, and general chatbots invent answers.
- Approach: hybrid retrieval, a citation-enforcing answer step, version-aware document handling, and a multi-agent orchestrator.
- Evaluation: [N] verified questions across [N] documents, with retrieval and answer quality metrics.
- Finding: [your strongest measured result, with its confidence interval].

## I. Introduction

- Why institutional documents are hard: scanned notices, tables, superseded circulars.
- Why grounding and refusal matter when the answer affects a student's exams or fees.
- Contributions, stated as claims you can back up:
  1. An ingestion pipeline for mixed-quality college documents, including version supersession.
  2. A controlled comparison of chunk size, embedding model and retrieval strategy on institutional text.
  3. An evaluation of answer faithfulness and refusal, with judge agreement against human labels.

## II. Related Work

- Retrieval-augmented generation: the original RAG formulation and later surveys.
- Hybrid retrieval and Reciprocal Rank Fusion.
- Cross-encoder re-ranking.
- Chunking strategies and their effect on retrieval.
- LLM evaluation: faithfulness and relevance metrics, and LLM-as-judge reliability.
- Educational chatbots and institutional knowledge assistants.
- Prompt injection in retrieval systems.

Read each paper you cite. Only cite what you have read.

## III. System Architecture

- Overview figure (from `docs/architecture.md`).
- Ingestion: extraction, OCR fallback, table handling, heading-aware chunking.
- Retrieval: dense and BM25 retrieval, RRF, filters, optional re-ranking.
- Answering: query rewriting, citations, refusal, guardrails.
- Agents: intent routing and multi-step execution.

## IV. Methodology

- Corpus: [N] documents by type, and how they were collected. State any permission needed.
- Test set: 150 questions across factual, multi-hop, unanswerable and time-sensitive. Who verified them, and how.
- Metrics: precision@5, recall@5, MRR, faithfulness, relevance, unanswerable accuracy, answerable coverage, p50 and p95 latency, cost per query.
- Judge validity: Cohen's kappa on 30 human-labelled answers.
- Statistics: paired bootstrap, 95% intervals.

## V. Experiments

1. Chunk size: 400, 800, 1200 tokens.
2. Embedding model: bge-small-en-v1.5 and all-MiniLM-L6-v2.
3. Retrieval: vector, hybrid, hybrid with re-ranking.
4. LLM: hosted and local.

Each varies one factor from the same baseline.

## VI. Results

- One table and one figure per experiment, from `eval/results/`.
- Results by question category. The overall average can hide what happened to multi-hop questions.
- Report what did not improve as plainly as what did.

## VII. Discussion

- Why the winning settings won, with example questions.
- Failure analysis: read 20 wrong answers, and group them by cause.
- Limits: one institution, one language, test set size, judge reliability.
- Threats to validity: questions written by the system's author, and the corpus shaping the tuning of thresholds.

## VIII. Conclusion and Future Work

- What was shown, in one paragraph.
- Next: multilingual documents, a user study with students, fine-tuned embeddings.

## Acknowledgements and references

Thank your guide and college. Use IEEE citation style.
