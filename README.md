# A Real-Time Gen-AI Customer Service Bot

One integrated customer-support platform covering six internship tasks: sentiment & escalation, ticketing & SLA,
knowledge-base pipeline, RAG assistant, multimodal (OCR) support and multilingual sessions.
Everything is deterministic and runs **offline** (no API key needed); time and configuration are injected so behaviour can be tested under simulated clocks.

## Setup (uv)
```bash
uv sync                      # installs dependencies from pyproject.toml / uv.lock
uv run pytest -q             # full test-suite (real-OCR tests run only if tesseract + pillow + pytesseract + pypdf are installed)
uv run a-real-time-gen-ai-customer-service-bot --cli     # chat in the terminal
uv run a-real-time-gen-ai-customer-service-bot           # HTTP API on http://127.0.0.1:8000/docs
```
Optional (for real screenshot/PDF OCR): `sudo apt install tesseract-ocr` and `uv add pillow pytesseract pypdf`.
Without them, image/PDF uploads fail gracefully ("OCR unavailable"); text files and all tests with fake OCR still work.

## Configuration
Everything tunable lives in `config/default.yaml` (+ `config/holidays.yaml`): business hours, working days, holidays, SLA minutes,
SLA counting mode, escalation thresholds, confidence thresholds, supported languages, session timeout/restore window,
retention, maintenance window, retry delays, quality-gate limits, roles and permissions. `Config.set()` / `Config.reload()` change values at runtime
(validated; an invalid change leaves the old value in place).

## API security
The caller never chooses its role. Set `BOT_API_KEYS` (see `.env.example`); the `X-API-Key` header maps to a role server-side.
No key = `customer`.

## Layout
```
config/                  YAML configuration
data/kb_docs/            sample knowledge base (versions, regions, access levels, expired/future/injected/restricted docs)
data/eval/               quality-gate evaluation cases
src/.../core/            Clock, Config, BusinessCalendar, masking, security (prompt injection), auth, logging
src/.../language/        entities, language detection, intents       (Task 6)
src/.../session/         session manager, summaries                  (Task 6)
src/.../sentiment/       sentiment + tone                            (Task 1)
src/.../escalation/      escalation engine                           (Task 1)
src/.../tickets/         extraction, priority, SLA, routing, service (Task 2)
src/.../kb_pipeline/     store, pipeline, scheduler                  (Task 3)
src/.../rag/             loader, retriever, answerer                 (Task 4)
src/.../multimodal/      validator, OCR, fields, storage, service    (Task 5)
src/.../monitoring/      metrics + alerts                            (Task 3)
src/.../orchestrator.py  the one entry point that connects everything
src/.../main.py          FastAPI app + CLI
tests/                   unit/, scenarios/ (hidden-test style), integration/
docs/                    architecture.md, requirements_matrix.md, report/
```
