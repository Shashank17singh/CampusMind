# Evaluation and research

Code: `eval/`. Steps are in [eval/README.md](../../eval/README.md).

## How to test

```bash
cd backend
pytest tests/test_eval.py
```

These tests cover the metrics, the test set rules, and report writing. They do not produce findings.

## Key decisions

**1. An unverified question cannot reach a result.** An LLM drafting a test set from the documents it will later be judged on is marking its own homework. Every draft is saved as `verified: false`, and `run_eval.py` refuses to run until a person has checked each one. `--allow-unverified` exists for trial runs, and anything produced that way is watermarked as synthetic.

**2. The LLM judge is checked against people.** A faithfulness score from an LLM means little on its own. Thirty answers are labelled by hand, before seeing the judge's scores, and Cohen's kappa measures agreement beyond chance. Plain percentage agreement would flatter a judge that gives nearly everything a 5.

**3. Differences are tested, not just compared.** With 150 questions, one system can beat another by luck. A paired bootstrap on per-question scores gives a 95% interval for each difference. If it contains zero, the honest statement is that the experiment found no clear difference.

## More design points

- **Why report coverage beside refusal accuracy?** A system that refuses everything scores 100% on unanswerable questions. Coverage shows the cost.
- **Why recall before precision?** The LLM cannot cite a passage that was never retrieved. Extra chunks cost tokens, but a missing one costs the answer.
- **Why one factor at a time?** Changing two things at once means a difference cannot be attributed to either.
- **Why is relevance judged per document, not per chunk?** Chunk ids change with chunk size, so per-chunk labels could not be compared across the experiment.
- **Why do charts use hatching and greyscale?** IEEE papers are often printed in black and white.

## Known limits

- Relevance per document is coarse. A retrieved document might contain the wrong passage.
- The questions are written by the system's author, so they may be easier than real student questions.
- A threshold tuned on the same questions used for reporting will look better than it is. Hold some questions back for tuning.
- `experiments.py` rewrites `chunks_meta`, so it needs its own database.
