# Requirement → implementation → test matrix

Status key: ✅ implemented and tested · 🟡 implemented with a known limit · 🔴 not implemented

"Tested" = an automated test in this repo passes (**916 passed, 1 skipped**; the skip is the "OCR not installed" fallback test,
which skips when the OCR libraries ARE installed). It does **not** mean the hidden evaluation will pass.

Quality measurements I made (on sets I wrote myself, so treat them as indicative, not as a guarantee):
* RAG paraphrase / relative-date benchmark: **14/34 before the hardening pass, 34/34 after** (`test_task4_paraphrase.py`)
* Sarcasm benchmark: **13/33 sarcastic sentences found before, 33/33 after; 30/30 genuine-praise sentences correctly left alone** (`test_task1_sarcasm_benchmark.py`)
* Fuzz: 3,000 random messages on 8 concurrent threads (random files, roles, languages, 4 simulated restarts): **0 handler errors, all ticket ids unique**
* **High-risk detection (account compromise, duplicate payment, legal threat), honest numbers.** Each fresh hold-out set was measured on its FIRST run, before I changed anything:
  hold-out 3 = 77% recall / 1 of 20 false positives; hold-out 4 = 69% / 1 of 25; hold-out 5 = 93% / 4 of 20. After fixing the general causes every set passes, **but those sets are now
  tuned, so they are regression tests, not proof.** Expect roughly **70-90% recall on unseen phrasing** from the rules alone. The optional LLM second opinion exists precisely for this gap.

## Task 1 – Sentiment, risk and escalation
| Requirement | Status | Code | Tests |
|---|---|---|---|
| six labels + confidence scores | ✅ | `sentiment/analyzer.py` | `test_sentiment_session.py` |
| current message + history | ✅ | `analyze(history=)` | `test_history_*` |
| tone adapts, policy never changes | ✅ | `sentiment/tone.py` | `test_tone_changes_prefix_but_never_the_body`, `test_policy_answers_do_not_depend_on_sentiment` |
| sarcasm (4 languages; praise next to a problem, strong idioms, emoji, praise after complaints; negation and "good news" guards) | ✅ rule-based | analyzer `_PRAISE`, `_NEG_SITUATION`, `_RESOLUTION`, `_NEGATION` | `test_task1_sarcasm_benchmark.py` (65 tests) |
| repeated negatives escalate | ✅ | `escalation/engine.py` | `test_task1_escalation.py` |
| account compromise / duplicate payment / legal threat, even when calm | ✅ rules (two layers: regexes + order-independent concept co-occurrence, with exclusions for how-to questions, hypotheticals, relatives, "twice as much"); 🟡 unseen phrasing ~70-90% | `language/intents.py` | `test_task1_high_risk_benchmark.py` (361 tests) |
| optional LLM second opinion for those three (adds labels only, never removes; untrusted-input prompt; failures ignored) | ✅ fake-client tests only; no real API call made | `language/llm_risk.py` | `test_task1_llm_risk.py` |
| > 15 min unresolved escalates automatically | ✅ (needs the background job or the next message) | `check_timeouts` | `test_unresolved_negative_*` |
| urgent after hours → on-call; normal → next working day | ✅ | `_make_record` | `test_*after_hours*` |
| reason, rule, timestamp, masked summary, metadata | ✅ | `EscalationRecord` | `test_record_has_*` |
| unseen sarcasm styles | 🟡 pattern based; a classifier would generalise further | | |

## Task 2 – Ticketing and SLA
| Requirement | Status | Code | Tests |
|---|---|---|---|
| extract customer/order/product/issue/evidence/contact, never invent | ✅ | `tickets/extractor.py` | `test_task2_tickets.py` |
| request missing mandatory information | ✅ | `missing_fields` | `test_missing_mandatory_*` |
| unresolved conversations → tickets (complaints, human-agent requests, **unanswerable questions after the customer accepts an offer**) | ✅ | orchestrator `offer_ticket` flow | `test_unanswerable_question_offers_a_ticket_*` |
| priority from severity, sentiment, waiting time, impact, SLA | ✅ (formula is my own design) | `tickets/priority.py` | `test_priority_*` |
| routing: skills, availability, workload, business hours | ✅ | `tickets/routing.py` | `test_routes_*`, `test_after_hours_*` |
| unavailable teams: fallback / backlog / re-route | ✅ | `Router` | `test_unavailable_team_*` |
| SLA excludes weekends and holidays | ✅ | `core/business_calendar.py` | calendar + SLA tests |
| 75 % warning; breach escalates automatically | ✅ | `tickets/sla.py` | `test_warning_at_75_*`, `test_breach_*` |
| runtime SLA changes | ✅ | config read per call | `test_runtime_*` |
| duplicates, grouping, separation | ✅ | `TicketService` | `test_duplicate_*`, `test_same_*`, `test_unrelated_*` |
| masked handoff summary, **available to agents via API and code** | ✅ | `GET /tickets/{id}/handoff`, `Orchestrator.ticket_handoff` | `test_ticket_and_handoff_endpoints`, `test_agents_can_read_tickets_*` |
| ticket persistence across restart | ✅ (JSON state file) | `state.py` | `test_restart_keeps_tickets_*` |
| duplicate detection of paraphrases / other languages | 🟡 word-overlap only | | |

## Task 3 – Knowledge-base pipeline
| Requirement | Status | Code | Tests |
|---|---|---|---|
| only new/modified documents processed | ✅ | `KBPipeline.ingest` | `test_only_new_or_modified_files_are_processed` |
| duplicates; quarantine invalid files | ✅ | `ingest`, `_quarantine` | `test_duplicate_*`, `test_invalid_files_*` |
| versions + rollback (manual, automatic) | ✅ | `kb_pipeline/store.py` | `test_versions_*`, `test_manual_rollback_*` |
| quality gate rejects accuracy/grounding drops | ✅ | `evaluate` | `test_update_that_reduces_accuracy_*`, `test_grounding_drop_*` |
| scheduled at configurable time; retries 15/30/60 | ✅ | `scheduler.py` | `test_failed_update_retries_*` |
| activation only in maintenance window | ✅ | `in_maintenance_window` | `test_window_*` |
| health checks 5 min; auto rollback; outage | ✅ default is a **smoke test** (live KB must answer sample questions); injectable | `smoke_health_check`, `health_tick` | `test_*health*`, `test_smoke_health_check_*` |
| access control (**attempts audited**: log + metric) | ✅ | `_authorize` | `test_unauthorised_*` |
| prompt-injection protection, masking | ✅ (also disguised: leetspeak, look-alike letters) | loader + `core/security.py` | `test_injection_in_document_*`, `test_disguised_injection_is_detected` |
| monitoring: latency, failures (**incl. failed updates, unauthorised access, quarantines**), confidence, escalations | ✅ metrics + alerts, no dashboard | `monitoring/metrics.py` | `test_metrics_*`, `test_failed_updates_and_giving_up_*` |
| KB version history across restart | 🔴 the KB is re-ingested from disk on start | | |
| SLA config must define every tier (a missing tier used to crash a ticket of that tier) | ✅ validated + defensive fallback | `core/config.py`, `tickets/sla.py` | `test_sla_config_must_define_every_tier` |

## Task 4 – RAG assistant
| Requirement | Status | Code | Tests |
|---|---|---|---|
| metadata (version, product, region, access, effective, expiry) | ✅ | `rag/loader.py` | loader tests |
| authorised retrieval only (before ranking) | ✅ | `rag/retriever.py` | `test_customer_cannot_retrieve_*` |
| latest applicable policy on conflict | ✅ | retriever | `test_latest_applicable_policy_wins_*` |
| current: ignore future and expired | ✅ | retriever | `test_expired_*`, `test_future_*` |
| historical: policy on the customer's date — **explicit date, relative ("last year", "6 months ago", "hace dos meses"), whole month; asks for the exact day when the policy changed inside that month** | ✅ | `language/entities.find_time_references`, answerer | `test_task4_dates_llm.py` |
| citation on every factual answer | ✅ | `citation_tag` | `test_every_answer_has_a_citation` |
| refuse / clarify on missing or ambiguous evidence (term-coverage rule: "free delivery" is not answered with shipping times) | ✅ | answerer, retriever | `test_missing_evidence_*`, paraphrase benchmark |
| unsupported-claim detection | ✅ | `verify_claims` | `test_unsupported_claim_detector` |
| ignore instructions in documents | ✅ (regression: legitimate text after an injected sentence is kept) | `sanitize_untrusted` | `test_injection_*`, `test_legitimate_sentence_after_an_injected_one_is_kept` |
| paraphrased questions | ✅ phrase normalisation + glossary (34-question benchmark) | `rag/query.py` | `test_task4_paraphrase.py` |
| LLM generation behind the verifier | ✅ `OpenAIGenerator` (prompt wraps evidence as untrusted; output verified; outage falls back). **Tested with a fake client only; no real API call was made** | `rag/llm.py` | `test_task4_dates_llm.py` |
| semantic (embedding) retrieval | 🔴 lexical + synonyms/glossary | | |

## Task 5 – Multimodal
| Requirement | Status | Code | Tests |
|---|---|---|---|
| text, screenshots, invoices, PDFs (**including image-only scanned PDFs**) | ✅ real Tesseract + pypdf verified | `multimodal/ocr.py` | `test_real_ocr.py` |
| extract order IDs, dates, amounts, product, error codes | ✅ | `fields.py` | `test_task5_multimodal.py` |
| compare with message **and conversation**; conflicts → ask (dates only compared when tied to a purchase/payment) | ✅ | `compare_with_message`, orchestrator `_purchase_dates` | `test_mismatched_*`, `test_unrelated_date_*` |
| low quality → another file; never invent | ✅ two OCR passes, ties favour the raw pass | `_analyze` | `test_blurred_*`, `test_noisy_rotated_photo_never_produces_invented_values` |
| mask PII/payment in logs | ✅ | masking + log filter | `test_summary_and_logs_never_contain_*` |
| unsafe files; hidden instructions in images (**incl. disguised text**) | ✅ | `validator.py`, `core/security.py` | `test_unsafe_files_*`, `test_hidden_instructions_*` |
| > 30 s → background queue + notify; **late result is merged into the conversation and the open ticket** | ✅ | `process`, `collect_background`, `_apply_attachment` | `test_slow_processing_*`, `test_late_background_result_*` |
| deleted after retention period (**also files left by a previous run; folder 0700, files 0600**) | ✅ | `FileStore.purge_expired`, `_adopt_existing` | `test_files_are_deleted_*`, `test_files_left_by_a_previous_run_*`, `test_upload_folder_and_files_are_private` |
| product photos | 🟡 a sharp, well-lit photo is accepted as **evidence** when the customer shows damage (sharpness, brightness, contrast, size checks); the bot does **not** recognise objects or damage | `imagequality.py` | `test_*product_photo*` |
| noisy phone photos | 🟡 unreadable results are never trusted; the customer is asked for a clearer file | | |

## Task 6 – Multilingual and sessions
| Requirement | Status | Code | Tests |
|---|---|---|---|
| 3 extra languages (hi, es, fr), mixed, switching | ✅ | `language/detector.py`, `i18n.py` | `test_language.py`, orchestrator tests |
| preserve names, order IDs, dates, product codes | ✅ | `language/entities.py` | `test_entities_preserved_exactly` |
| ≥ 10 messages of context; isolated sessions | ✅ | `session/` | `test_context_keeps_at_least_ten_messages`, `test_sessions_are_isolated_*` |
| low language/intent confidence → clarify (**no language question for "ok"**) | ✅ | orchestrator | `test_low_confidence_input_*`, `test_trivial_first_message_*` |
| spelling errors, transliteration, multiple requests, corrections | ✅ (finite vocabulary) | `intents.normalize`, `merge_entities` | language + orchestrator tests |
| **follow-ups use context** ("and in India?", "what about the refund after a return?"); empty follow-ups ask which one | ✅ | orchestrator `_FOLLOWUP` | `test_*followup*` |
| 30 min inactivity ends session; ≤ 24 h restore summary; else new | ✅ **also across a restart** (summary + non-sensitive entities, masked) | `session/manager.py`, `state.py` | `test_exact_boundaries`, `test_restart_keeps_*` |
| languages, thresholds, timings configurable | ✅ | `config/default.yaml` | config + session tests |
| unseen multilingual conversations | 🟡 rule-based coverage is finite (fuzz-tested for crashes only) | | |

## Still true after this pass
* No real LLM or API call has been exercised; only the hook and its safety checks are tested.
* NLP is rule based (explainable, offline). New slang, new sarcasm styles and further languages need new patterns.
* Metrics windows, the KB version history and uploaded files are not persisted.
* Evaluate against the portal's instructions: anything this matrix does not mention is not covered.

## Added by the full-project audit
| Finding | Fix | Test |
|---|---|---|
| Typo corrector rewrote real words into risk words (**police→policy, changes→charges, charger→charges**) causing missed and false escalations | stricter fuzzy cutoff (0.88) + protected word list; high-risk layer tests corrected AND original text | `test_typo_corrector_never_rewrites_real_words_*`, hold-out sets |
| No lock around shared state under concurrent requests | one re-entrant lock, released during slow OCR | `test_simultaneous_customers_*`, `test_a_slow_upload_does_not_block_other_customers` |
| Upload folder world-readable; files from a previous run never deleted (retention violation) | 0700/0600 + adoption of leftovers | `test_files_left_by_a_previous_run_*` |
| Metrics list grew forever | bounded + pruned | `test_metrics_do_not_grow_without_bound` |
| `.env` documented but never loaded | `load_env_file()` in the API and CLI | `test_dotenv_file_is_actually_loaded` |
| Unvalidated language codes could be configured | validated against implemented languages | `test_supported_languages_must_be_implemented` |
| Unbounded message / conversation size | 8,000-char cap, 500-message cap | `test_extremely_long_message_*`, `test_very_long_conversations_are_bounded` |
| API key compared with a normal dict lookup | constant-time comparison | `test_resolve_role_matches_exactly` |
| Background job blocked the event loop; OCR workers never stopped | `asyncio.to_thread`, `Orchestrator.close()` | `test_close_is_safe_to_call_twice` |
| Two assertions could never fail (`a and b or c` precedence; `x == [] or ...`) | rewritten so they can fail | (fixed in place) |
