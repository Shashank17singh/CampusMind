# Study planner

Code: `backend/app/agents/scheduler.py`, `backend/app/agents/planner.py`, `backend/app/api/study_plans.py`.

## How to run

```bash
curl -X POST localhost:8000/api/v1/study-plans \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"exams": [{"subject": "DBMS", "date": "2026-12-10"}],
       "hours_per_weekday": {"0": 3, "1": 3, "2": 3, "3": 3, "4": 3, "5": 5, "6": 5},
       "difficulty": {"DBMS": 4}}'
```

## How to test

```bash
pytest tests/test_planner.py
```

`test_rules_hold_on_random_inputs` runs the hard rules on 25 seeded random plans. This matters more than any hand-picked case, because it tries combinations no one would think to write.

## Why scheduling is done in code, not by the LLM

An LLM predicts plausible text. It does not calculate. Asked for a plan, it will produce something that looks right, and may also put six hours on a three-hour day, or schedule a new topic the night before an exam. Nothing in the output would reveal the mistake, and a student could lose marks by trusting it.

| Need | Code | LLM |
|---|---|---|
| Never exceed daily hours | Guaranteed | Probable at best |
| Same input gives same plan | Always | No |
| Cost per plan | Free | Tokens every time |
| Re-plan instantly | Yes | Another slow call |
| Can be unit tested | Yes, including random inputs | Only by sampling |
| Can be explained step by step | Yes | No |

The LLM is used for what it is good at: reading an unstructured syllabus into structured topics. So each part does the job it can do reliably. The LLM's output is a list of names, validated by a schema, and a wrong topic name is visible and harmless. A wrong schedule would not be.

## Key decisions

**1. The scheduler is a pure function.** It takes topics, exams and hours, and returns slots, with no database or LLM. That is why random-input testing is possible, and why a plan can be regenerated identically.

**2. Greedy earliest-fit with a priority of difficulty over days left.** Harder topics with closer exams go first, each into the earliest free time. Learning uses at most 75% of a day, which keeps room for later spaced revisions. *Rejected:* an optimal solver such as integer programming. It would give slightly better plans, but with a heavy dependency, for a problem whose inputs (a student's real hours) are only estimates. Greedy is fast, easy to explain, and good enough here.

**3. Re-planning rebuilds only pending work.** Done items are never touched. Unfinished pending items from the restart date are deleted and scheduled again, and items already past become `missed`. A topic counts as learned only once its full learning time is done, and its revisions are still scheduled from the date it was finished. The original inputs and resolved topics are stored on the plan, so re-planning needs no LLM.

## How the rules are implemented

| Rule | Implementation |
|---|---|
| Spaced repetition | Revise at 1, 3 and 7 days, moving one day later if that day is full |
| Revision only before exams | The two days before any exam never get `learn` items |
| Never exceed hours | Every placement subtracts from that day's remaining minutes |
| Practice slots | One past-papers hour per subject, on its first revision day |

## Known limits

- Two days before any exam means revision only for all subjects. With exams every three days, nothing new can be learned between them. Students may find this strict.
- Sessions start at 09:00, because the plan takes hours per day, not preferred times.
- If a topic cannot fit, it is reported and left out, not squeezed in. The user decides what to give up.
- Syllabus text past 24,000 characters is cut off before extraction, and this is logged.
