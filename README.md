# A Real-Time Gen-AI Customer Service Bot

One integrated customer-support platform covering six internship tasks: sentiment & escalation, ticketing & SLA,
knowledge-base pipeline, RAG assistant, multimodal (OCR) support and multilingual sessions.
Everything is deterministic and runs **offline** (no API key needed); time and configuration are injected so behaviour can be tested under simulated clocks.

## Setup (uv)
```bash
uv sync                      # installs dependencies from pyproject.toml / uv.lock
uv sync --extra ocr          # + real screenshot/PDF OCR libraries (needs the tesseract binary)
uv sync --extra llm          # + openai client (only if you enable the LLM, see below)
uv run pytest -q             # full test-suite (real-OCR tests run only if tesseract + pillow + pytesseract + pypdf are installed)
uv run a-real-time-gen-ai-customer-service-bot --cli     # chat in the terminal
uv run a-real-time-gen-ai-customer-service-bot           # HTTP API on http://127.0.0.1:8000/docs
```
Optional (real screenshot/PDF OCR): install the `tesseract` binary with your package manager, then `uv sync --extra ocr`.
Without them, image/PDF uploads fail gracefully ("OCR unavailable"); text files and all tests with fake OCR still work.

## Optional features (environment variables, see `.env.example`)
| Variable | Effect |
|---|---|
| `BOT_API_KEYS` | JSON `{"key": "role"}`; the role comes from the `X-API-Key` header, never from the request body |
| `BOT_STATE_FILE` | path of the JSON state file (tickets, escalations, returning-customer sessions); loaded on start, saved every minute and on shutdown, owner-only permissions |
| `BOT_USE_LLM=1` + `OPENAI_API_KEY` | (1) answers are phrased by an LLM, then **verified sentence by sentence against the evidence**; unsupported output falls back to the safe extractive answer. (2) an LLM **second opinion for high-risk messages** (account compromise, duplicate payment, legal threat) that can only ADD escalations, never remove one |
| `BOT_LLM_MODEL` | model name for the above (default `gpt-4o-mini`) |

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

## Deployment notes (read before exposing the API)
* **Customer identity is trusted from the caller.** `customer_id` is whatever the client sends, so put the API behind your own authenticated gateway
  that sets it. Anyone who can call `/chat` or `/notifications/{id}` with someone else's id acts as that customer.
* Roles come only from server-side API keys (`BOT_API_KEYS`, compared in constant time). Keys and the state file are secrets; the state file holds
  customer contact details, so keep it on an encrypted volume. Uploads are stored under random names with owner-only permissions and are deleted after
  the retention period, **including files left by a previous run**.
* One process-wide lock protects shared state, so simultaneous customers cannot corrupt ids or counters; it is released while a slow upload is being
  read, so one customer's OCR never freezes the others. Scale out with one process per instance plus an external store (the JSON state file is single-writer).
* Background work (15-minute escalation, SLA warnings, KB updates, file purge) runs every minute in the API process.
* There is no rate limiting; add it at the gateway.
