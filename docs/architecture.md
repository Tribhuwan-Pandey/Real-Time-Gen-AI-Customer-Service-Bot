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

## Added in the hardening pass
* **Follow-ups use context**: `conv.last_question` lets "and in India?" or "what about the refund after a return?" continue the previous knowledge question; a follow-up with no new content asks which one is meant.
* **Unanswerable questions can become tickets**: the bot offers ("Would you like me to open a ticket?"); a yes (any supported language) creates the ticket with the original question as the issue and asks only for what is missing.
* **Time references**: explicit dates, relative dates ("last year", "6 months ago", "hace dos meses") and whole months. If the policy changed inside the month the bot asks for the exact day. Historical answers start with "As of YYYY-MM-DD:".
* **Answerability rule**: besides the idf score, the evidence must contain a minimum share of the question's content words (`rag.min_term_coverage`), so "free delivery" is not answered with shipping times.
* **Dates in invoices**: only dates the customer wrote next to purchase/payment words are compared with the invoice.
* **Late OCR results** are merged into the conversation and any open ticket (as evidence) and delivered through notifications.
* **Product photos** with no text are accepted as evidence only when the customer is showing damage/defect and the picture passes sharpness/brightness/contrast/size checks.
* **Persistence** (`state.py`): tickets, escalations, pending notifications and a masked summary + non-sensitive entities per returning customer are saved to one JSON file (atomic write, mode 0600).
* **Audit**: denied KB and ticket actions are logged and counted (`unauthorized_access`); failed and abandoned KB updates are counted too.

## Added by the full-project audit
* **High-risk intents** use two layers: extra regexes and an order-independent *concept* layer (something of mine + an unauthorised/unknown marker; a money word + a "twice" marker; a legal word + a future-intent marker), evaluated per sentence and per pair of adjacent sentences. Exclusions (how-to questions, hypotheticals, relatives/third parties, "twice as much", "lawyer-approved") apply to both layers.
* **Typo correction is deliberately conservative** (cutoff 0.88, protected words) because a wrong correction can create or destroy an escalation.
* **Concurrency**: one re-entrant lock around shared state; released while a slow upload is read.
* **Retention** covers files left by a previous run; **memory** is bounded (metrics, message history, input length).

## Known limitations
* The **knowledge-base version history, metrics windows and uploaded files are not persisted** (the KB is re-ingested from `data/kb_docs` on start).
* Sentiment, language and intent detection are **rule based**. They are explainable and testable but will miss unseen slang, new sarcasm styles and further languages. A classifier can be placed behind the same interfaces.
* Retrieval is **lexical** (+ phrase normalisation and a small es/fr/hi glossary). KB documents are English; answers quote the English text. Embeddings would be the next step.
* **No real LLM call was made.** The `OpenAIGenerator` is wired and its safety behaviour (untrusted wrapping, claim verification, outage fallback) is tested with a fake client only.
* The bot does **not recognise objects or damage in photos**; it only checks photo quality and keeps the photo as evidence. Very noisy phone photos are usually unreadable, and then the customer is asked for a clearer file (values are never guessed).
* Duplicate-ticket detection compares words; paraphrases or other languages of the same complaint may become separate (grouped or ungrouped) tickets.
* The scheduler and background checks need something to call them (the API does, every minute).
* High-risk detection by rules alone generalises to roughly 70-90% of unseen phrasings; enable the optional LLM second opinion for more.
* `customer_id` is trusted from the caller (put an authenticated gateway in front); there is no rate limiting.
