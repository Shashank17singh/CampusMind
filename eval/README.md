# Evaluation

Tools for measuring CampusMind, and for producing the tables and charts in the report and paper.

**No result in this repository is a finding yet.** `eval/results/synthetic-smoke/` only shows that the tools run, on invented documents. Real results come from the corpus below, once its questions have been checked by a person.

## The corpus

[corpus/lucknow.json](corpus/lucknow.json) lists 34 public English documents from the University of Lucknow, centred on B.Tech. at the Faculty of Engineering and Technology:

- B.Tech. ordinances (2018, 2023 and the 2026 revision), the undergraduate ordinance and university policies
- Syllabi for every year of CSE (AI), and for CSE, ECE, ME, EE and Civil
- Admission notices, the 2026-27 admission brochure and student guides

The PDFs are not in the repository. `fetch_corpus.py` downloads them and checks each one's SHA-256, so a document the university has since replaced is reported instead of silently changing the questions' answers.

Most of these documents are scans, so ingestion needs Tesseract. Hindi notices, which include every examination notice and the academic calendar, are left out, because the embedding model reads English only. Say so when reporting results.

## Steps

Run the Python steps from `backend/`, against a database used only for evaluation, because `experiments.py` rewrites chunks.

1. Download the corpus: `python ../eval/fetch_corpus.py ../eval/corpus/lucknow.json`
2. Ingest it, OCR included: `python ../eval/ingest_corpus.py ../eval/corpus/lucknow.json`. This writes `lucknow.ids.json`, which maps each file to its document id.
3. Draft questions, a few per document, spread through each one: `python ../eval/draft_testset.py --output ../eval/data/testset_draft.json --per-document 2`
4. Make a review sheet: `python ../eval/review_sheet.py export ../eval/data/testset_draft.json ../eval/data/review.csv --ids ../eval/corpus/lucknow.ids.json`
5. **Check every row.** Open `review.csv` in Excel or Google Sheets, look the answer up at the place in `source`, correct it if needed, and put `y` in `keep`. Add multi-hop rows by hand, naming each document's file, separated by semicolons. About 60 checked questions across the four categories is enough. 150 is better, but a smaller set checked by one person is honest, as long as the paper says so.
6. Read the sheet back: `python ../eval/review_sheet.py import ../eval/data/review.csv ../eval/data/testset.json --ids ../eval/corpus/lucknow.ids.json`
7. Label 30 answers by hand in `eval/data/human_labels.json`, as `[{"question": "...", "faithfulness": 4}]`, using the rubric in `run_eval.py`. Label before seeing the judge's scores.
8. Run the retrieval experiments: `python ../eval/experiments.py ../eval/data/testset.json`
9. Run the full evaluation once per LLM, changing `LLM_PROVIDER`:

   ```bash
   python ../eval/run_eval.py ../eval/data/testset.json --name groq --human-labels ../eval/data/human_labels.json
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
