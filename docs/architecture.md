# Architecture

## One request, all modules
`Orchestrator.handle_message(customer_id, text, role, region, attachments)`

1. **Session** (`SessionManager`): get/create the customer's isolated conversation; expire after 30 min idle; restore the summary if the customer returns within 24 h.
2. **Untrusted input**: instruction-like sentences are stripped (`sanitize_untrusted`); never obeyed.
3. **Language** (`LanguageDetector`): en / hi (Devanagari + romanised) / es / fr, mixed-language aware; low confidence falls back to the session language, unsupported languages are declined politely.
4. **Entities**: order IDs, dates, product codes, amounts, error codes, emails, phones, names kept *exactly as written*; corrections ("sorry, I meant ORD-222") replace old values.
5. **Sentiment + intents**: six labels with confidence, using conversation history; multilingual intents with typo tolerance; several intents per message.
6. **Attachments** (`MultimodalService`): validate -> store -> OCR (30 s budget, else background queue) -> quality check -> hidden-instruction check -> extract -> compare with what the customer said.
7. **Escalation** (`EscalationEngine`): high-risk intents (even if calm), repeated negatives, unresolved > 15 min, after-hours routing (on-call vs scheduled).
8. **Answer**: knowledge question -> `KnowledgeAssistant` (authorised, date-valid, latest-policy, cited, verified); complaint -> `TicketService` (missing info is requested, never invented).
9. **Tone**: only a courteous prefix depends on sentiment. The business body is identical for calm and angry customers (tested).

`run_background_checks()` (every minute in the API) handles what must happen without a message: the 15-minute sweep, SLA warnings/breaches, re-routing,
session expiry, file purge, scheduled KB update, health check / auto-rollback, and background-OCR notifications.

## Cross-cutting design
| Concern | Where |
|---|---|
| Time is injected | `core/clock.py` (`SystemClock`, `FakeClock`); nothing calls `datetime.now()` |
| Nothing hard-coded | `config/*.yaml`, read on every call, validated, runtime-changeable |
| Weekends/holidays/hours | `core/business_calendar.py` |
| Authorisation in retrieval | `rag/retriever.py` filters by role *before* ranking (restricted text never reaches scoring) |
| Masking | `core/masking.py`, used by logs, summaries, handoffs, ingestion |
| Prompt injection | `core/security.py`; documents, OCR text and messages are data |

## Interpretations I had to make (state these in the report / viva)
* **SLA counting**: `sla.count_mode` = `calendar` (default; 24 h counted on working days, weekends/holidays excluded, the literal reading of the spec) or `business_hours`.
* **SLA tier is fixed at ticket creation**; the dynamic priority (which rises with waiting time) is used for ordering/routing. Otherwise a ticket's own waiting time would shorten its deadline and cause spurious breaches.
* **Warning at >= 75 %**, **breach when consumed > allowed** (exactly at the deadline is still inside the SLA).
* **"Unresolved for more than 15 minutes"** is strict (> 15). **Session timeout** is `>=` 30 min idle. **Restore** is `<=` 24 h. **File retention** deletes at `>=` the period.
* **Document expiry date is the last valid day (inclusive).**
* **"Next working day"** for after-hours normal complaints = next opening time (so Saturday -> Monday 09:00, Monday 02:00 -> Monday 09:00).
* **Retry delays** 15 / 30 / 60 min are measured from the previous failure.
* **Conflicting regional policies**: latest effective date wins; if the region is unknown and variants exist, the bot asks for the region.
* **Roles** come from server-side API keys, never from the request body.

## Known limitations
* State (sessions, tickets, escalations, KB versions) is **in memory**: a restart loses it. Session restore works within one running process. A database layer would be the next step.
* Sentiment, language and intent detection are **lexicon/rule based** (offline, explainable, testable). They will miss unseen slang or sarcasm; an LLM/embedding classifier could be plugged in behind the same interfaces.
* Retrieval is **lexical** (+ a small es/fr/hi glossary). KB documents are English; answers quote the English KB text.
* Product *photos*: only text visible in an image is read (OCR); no visual object recognition. Scanned PDFs are not OCR'd (the bot asks for an image/other file).
* `generator=` accepts an LLM, whose output is verified against evidence (tested with fake generators only; no real LLM was called).
