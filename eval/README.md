# Evaluation

Tools for measuring CampusMind, and for producing the tables and charts in the report and paper.

**No result in this repository is a finding yet.** There are no real college documents here, so anything under `eval/results/synthetic-smoke/` only shows that the tools run. Rerun everything on real documents before reporting a number.

## Steps

Run these from `backend/`, against a database used only for evaluation, because `experiments.py` rewrites chunks.

1. Ingest your real documents.
2. Draft questions: `python ../eval/draft_testset.py --output ../eval/data/testset_draft.json`
3. **People verify every question.** Check the answer against the document, fix or delete it, write multi-hop questions by hand, and set `"verified": true`. Aim for 150 questions across all four categories. Save the file as `eval/data/testset.json`.
4. Label 30 answers by hand in `eval/data/human_labels.json`, as `[{"question": "...", "faithfulness": 4}]`, using the rubric in `run_eval.py`. Label before seeing the judge's scores.
5. Run the retrieval experiments: `python ../eval/experiments.py ../eval/data/testset.json`
6. Run the full evaluation once per LLM, changing `LLM_PROVIDER`:

   ```bash
   python ../eval/run_eval.py ../eval/data/testset.json --name hosted --human-labels ../eval/data/human_labels.json
   python ../eval/run_eval.py ../eval/data/testset.json --name local --human-labels ../eval/data/human_labels.json
   ```

`run_eval.py` spends money with a hosted LLM: one answer and one judge call per question, plus query rewrites and follow-ups.

## What is measured

| Metric | How |
|---|---|
| Precision@5, recall@5, MRR | Per document, in `retrieval_bench.py` |
| Faithfulness, relevance | An LLM judge with a fixed rubric, in `run_eval.py` |
| Judge agreement | Cohen's kappa against 30 human labels. Under about 0.6, do not rely on the judge |
| Unanswerable detection | Share of unanswerable questions correctly refused |
| Answerable coverage | Share of answerable questions not refused. Report this beside refusal accuracy, because a system that refuses everything scores perfectly on refusals |
| Latency | p50 and p95 per question |
| Cost per query | From recorded tokens and the price table |
| Significance | A paired bootstrap on per-question scores |

## Output

`eval/results/` gets `experiments.csv`, `experiments.md`, and a PNG and PDF chart per experiment.

The paper outline is in [paper_outline.md](paper_outline.md).
