# Internship report: material to write from (do NOT paste as-is)

The report must be in your own words and describe only what you did and verified. Required sections and the *facts* you can draw on:

1. **Introduction** – project: integrated AI customer-support platform; six tasks implemented as modules of one system.
2. **Background** – course base application; gap: sentiment, ticketing, KB operations, RAG, multimodal, multilingual sessions.
3. **Learning objectives** – you decide; examples: dependency-injected time/config for testability, retrieval-time authorisation, prompt-injection handling, SLA arithmetic over calendars.
4. **Activities and tasks** – per-task summary: see `docs/architecture.md` and `docs/requirements_matrix.md`.
5. **Skills and competencies** – Python packaging with uv, pytest (526 tests), FastAPI, YAML config, Tesseract OCR, regex/NLP rules, state machines (scheduler, rollback).
6. **Feedback and evidence** – test output (`uv run pytest -v` screenshot), demo conversations, fuzz run (1,500 random messages, 0 crashes), git history.
7. **Challenges and solutions** – real examples found during development (write what *you* encountered):
   * SLA deadline shortened by the ticket's own rising priority → fixed SLA tier at creation.
   * KB update re-evaluated after threshold change was wrongly skipped → fingerprint includes config.
   * Rollback overwrote the status of the failed version → ordering bug fixed.
   * Region ambiguity flagged for the wrong policy → scoped to the matched policy.
   * Sanitiser marker leaked into search terms; weak one-word matches triggered clarification.
   * **Measured first, then fixed**: a 34-question paraphrase benchmark passed only 14/34 at first; after phrase normalisation, month/relative-date handling and a term-coverage rule it passes 34/34. A 33-sentence sarcasm benchmark went from 13/33 to 33/33; a stress test with genuine messages then showed 8/20 false alarms ("I'll order another one, thanks!"), which led to negation and "good news" guards.
   * The sanitiser's replacement marker had no full stop, so it glued itself to the next legitimate sentence and the answer builder discarded both: real content after an injected sentence was being lost.
   * OCR "cleaning" improved confidence but sometimes truncated digits; the fix was to trust the raw pass on ties.
   * A one-line "other" in a region-word table made "and the other one?" re-answer the previous question.
   * **Measured honestly**: fresh hold-out sets for the three high-risk intents first scored 77%, 69% and 93% recall (with up to 4 of 20 false positives) before any fix; after fixing the general causes they pass, but they are then tuned data, so I report the first-run numbers as the realistic estimate (about 70-90% on unseen phrasing).
   * The typo corrector turned "police" into "policy" and "changes" into "charges": a wrong correction can create or destroy an escalation, so correction is now conservative and protected.
   * Self-review found: no lock for concurrent customers, uploads from a previous run never deleted (retention), a world-readable upload folder, an unbounded metrics list, a documented `.env` that was never loaded, and two test assertions that could never fail.
   * I also lost an unsaved working copy once and had to rebuild from the last delivered zip: lesson, package after every stage.
   * Monitoring gap found during a self-review: failed KB updates and unauthorised attempts were not counted.
8. **Outcomes and impact** – what works (matrix) and honest limitations (in-memory state, rule-based NLP).
9. **Conclusion** – lessons learnt and next steps (database persistence, embeddings, LLM generator with verification).

Reminders from your internship terms: fill the daily report form every day; the submission is reviewed once; work must be original and you must be able to explain it.
