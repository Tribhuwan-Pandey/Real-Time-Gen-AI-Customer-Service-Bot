# Internship report: material to write from (do NOT paste as-is)

The report must be in your own words and describe only what you did and verified. Required sections and the *facts* you can draw on:

1. **Introduction** – project: integrated AI customer-support platform; six tasks implemented as modules of one system.
2. **Background** – course base application; gap: sentiment, ticketing, KB operations, RAG, multimodal, multilingual sessions.
3. **Learning objectives** – you decide; examples: dependency-injected time/config for testability, retrieval-time authorisation, prompt-injection handling, SLA arithmetic over calendars.
4. **Activities and tasks** – per-task summary: see `docs/architecture.md` and `docs/requirements_matrix.md`.
5. **Skills and competencies** – Python packaging with uv, pytest (360 tests), FastAPI, YAML config, Tesseract OCR, regex/NLP rules, state machines (scheduler, rollback).
6. **Feedback and evidence** – test output (`uv run pytest -v` screenshot), demo conversations, fuzz run (1,500 random messages, 0 crashes), git history.
7. **Challenges and solutions** – real examples found during development (write what *you* encountered):
   * SLA deadline shortened by the ticket's own rising priority → fixed SLA tier at creation.
   * KB update re-evaluated after threshold change was wrongly skipped → fingerprint includes config.
   * Rollback overwrote the status of the failed version → ordering bug fixed.
   * Region ambiguity flagged for the wrong policy → scoped to the matched policy.
   * Sanitiser marker leaked into search terms; weak one-word matches triggered clarification.
8. **Outcomes and impact** – what works (matrix) and honest limitations (in-memory state, rule-based NLP).
9. **Conclusion** – lessons learnt and next steps (database persistence, embeddings, LLM generator with verification).

Reminders from your internship terms: fill the daily report form every day; the submission is reviewed once; work must be original and you must be able to explain it.
