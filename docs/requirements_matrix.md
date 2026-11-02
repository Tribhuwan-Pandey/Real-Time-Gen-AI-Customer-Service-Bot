# Requirement → implementation → test matrix

Status key: ✅ implemented and tested · 🟡 partially implemented / limited · 🔴 not implemented · ⚠️ implemented, weakly tested

"Tested" means an automated test in this repo passes (360 passed, 1 skipped at the time of writing). It does **not** mean it will pass the hidden evaluation.

## Task 1 – Sentiment, risk and escalation
| Requirement | Status | Code | Tests |
|---|---|---|---|
| positive / neutral / negative / frustrated / urgent / sarcastic + confidence scores | ✅ (rule-based; unseen phrasing may be missed) | `sentiment/analyzer.py` | `test_sentiment_session.py` |
| uses conversation history | ✅ | `analyzer.analyze(history=)` | `test_history_*` |
| tone adapts, business policy never changes | ✅ | `sentiment/tone.py`, orchestrator | `test_tone_changes_prefix_but_never_the_body`, `test_policy_answers_do_not_depend_on_sentiment` |
| repeated negatives escalate (N configurable) | ✅ | `escalation/engine.py` | `test_task1_escalation.py` |
| account compromise / duplicate payment / legal threat escalate, even when calm | ✅ | intents + engine | `test_calm_high_risk_escalates` (5 languages) |
| > 15 min unresolved negative escalates automatically | ✅ (needs `run_background_checks`/message) | `check_timeouts` | `test_unresolved_negative_*` |
| urgent after hours → on-call; normal → next working day | ✅ | `EscalationEngine._make_record` | `test_*after_hours*` |
| record reason, rule, timestamp, summary, metadata | ✅ | `EscalationRecord` | `test_record_has_reason_rule_timestamp_summary_metadata` |
| sarcasm, calm high-risk, after-hours, simulated time | ✅ / 🟡 sarcasm is pattern based | | scenario tests |

## Task 2 – Ticketing and SLA
| Requirement | Status | Code | Tests |
|---|---|---|---|
| extract customer/order/product/issue/evidence/contact; never invent | ✅ | `tickets/extractor.py` | `test_task2_tickets.py`, integration |
| request missing mandatory info (configurable list) | ✅ | `missing_fields` | `test_missing_mandatory_*` |
| priority from severity, sentiment, waiting time, impact, SLA | ✅ (formula is my own design) | `tickets/priority.py` | `test_priority_*` |
| routing by skill, availability, workload, business hours | ✅ | `tickets/routing.py` | `test_routes_*`, `test_after_hours_*` |
| unavailable teams: fallback / backlog + supervisor flag + re-route | ✅ | `Router.route/reroute_waiting` | `test_unavailable_team_*` |
| SLA excludes weekends and holidays (configurable) | ✅ | `core/business_calendar.py` | calendar + SLA tests |
| 75 % warning, breach escalates automatically | ✅ | `tickets/sla.py` | `test_warning_at_75_*`, `test_breach_*` |
| runtime SLA / schedule changes | ✅ | config read per call | `test_runtime_*` |
| duplicate detection, grouping, separating unrelated | ✅ | `TicketService._find_duplicate_and_related` | `test_duplicate_*`, `test_same_*`, `test_unrelated_*` |
| masked handoff summary | ✅ | `handoff_summary` | `test_handoff_summary_masks_*` |
| persistence of tickets | 🔴 in-memory only | | |

## Task 3 – Knowledge-base pipeline
| Requirement | Status | Code | Tests |
|---|---|---|---|
| only new/modified documents processed | ✅ | `KBPipeline.ingest` | `test_only_new_or_modified_files_are_processed` |
| duplicates detected, invalid files quarantined (with reason) | ✅ | `ingest`, `_quarantine` | `test_duplicate_*`, `test_invalid_files_*` |
| versions + rollback (manual and automatic) | ✅ | `kb_pipeline/store.py` | `test_versions_*`, `test_manual_rollback_*` |
| quality gate: reject accuracy / grounding drops | ✅ | `KBPipeline.evaluate` | `test_update_that_reduces_accuracy_*`, `test_grounding_drop_*` |
| scheduled at configurable time; retries 15/30/60 | ✅ | `kb_pipeline/scheduler.py` | `test_failed_update_retries_*` |
| activation only in maintenance window | ✅ | `in_maintenance_window` | `test_window_*`, `test_approved_update_outside_window_*` |
| health checks 5 min, auto rollback on failure/outage | ✅ (default health check is simple: index non-empty; injectable) | `health_tick` | `test_*health*`, `test_service_outage_*` |
| access control | ✅ | `AuthorizationService` | `test_unauthorised_roles_cannot_update` |
| prompt-injection protection, sensitive-data masking | ✅ | loader sanitises + masks | `test_injection_in_document_*`, `test_pii_is_masked_*` |
| monitoring: latency, failures, confidence, escalations | 🟡 metrics + alerts exist; no dashboard / external exporter | `monitoring/metrics.py` | `test_metrics_*` |

## Task 4 – RAG assistant
| Requirement | Status | Code | Tests |
|---|---|---|---|
| metadata: version, product, region, access level, effective & expiry date | ✅ | `rag/models.py`, `loader.py` | loader tests |
| authorisation enforced inside retrieval | ✅ | `rag/retriever.py` | `test_customer_cannot_retrieve_*`, `test_restricted_docs_never_reach_ranking_*` |
| latest applicable policy on conflict | ✅ | retriever conflict step | `test_latest_applicable_policy_wins_*` |
| current: ignore future and expired; historical: policy active on that date | ✅ | retriever + answerer | `test_expired_*`, `test_future_*`, `test_historical_*` |
| citation on every factual answer | ✅ | `citation_tag` | `test_every_answer_has_a_citation` |
| refuse / clarify when evidence missing or ambiguous | ✅ | answerer | `test_missing_evidence_*`, `test_ambiguous_*`, region test |
| unsupported-claim detection | ✅ | `verify_claims` | `test_unsupported_claim_detector`, generator tests |
| ignore malicious instructions in documents | ✅ | `sanitize_untrusted` at load and answer time | `test_injection_*` |
| semantic (embedding) retrieval | 🔴 lexical + glossary only | | |
| real LLM generation | 🔴 hook exists (`generator=`), tested with fakes only | | |

## Task 5 – Multimodal
| Requirement | Status | Code | Tests |
|---|---|---|---|
| text, screenshots, invoices, PDFs | ✅ real Tesseract + pypdf verified in this environment | `multimodal/ocr.py` | `test_real_ocr.py` |
| extract order IDs, dates, amounts, product, error codes | ✅ | `fields.py` | `test_task5_multimodal.py` |
| compare with message; conflicts → ask | ✅ | `compare_with_message` | `test_mismatched_*` |
| low quality → ask for another file; never invent | ✅ | service `_analyze` | `test_blurred_*`, `test_missing_values_are_never_invented` |
| mask PII/payment in logs | ✅ | masking + log filter | `test_summary_and_logs_never_contain_*` |
| reject unsafe files; hidden instructions in images | ✅ | `validator.py`, injection check | `test_unsafe_files_*`, `test_hidden_instructions_*` |
| > 30 s → background queue + notify | ✅ (threads; timeout configurable) | `MultimodalService.process`, `collect_background` | `test_slow_processing_*` |
| deleted after retention period | ✅ | `FileStore.purge_expired` | `test_files_are_deleted_*`, `test_expired_file_*` |
| product *photo* understanding (objects, damage) | 🔴 only text inside images is read | | |
| scanned (image-only) PDFs | 🟡 treated as unreadable → asks for another file | | |

## Task 6 – Multilingual and sessions
| Requirement | Status | Code | Tests |
|---|---|---|---|
| 3 additional languages (hi, es, fr), mixed, switching | ✅ | `language/detector.py`, `i18n.py` | `test_language.py`, `test_reply_language_follows_customer_*` |
| preserve names, order IDs, dates, product codes | ✅ | `language/entities.py` | `test_entities_preserved_exactly` |
| ≥ 10 messages of context; isolated sessions | ✅ | `session/` | `test_context_keeps_at_least_ten_messages`, `test_sessions_are_isolated_*` |
| low language/intent confidence → clarify | ✅ | orchestrator | `test_low_confidence_input_*` |
| spelling errors, transliteration, multiple requests, corrections | ✅ (limited vocabulary) | `intents.normalize`, `merge_entities` | corresponding tests |
| 30 min inactivity ends session; ≤ 24 h restore summary; else new | ✅ in-process | `session/manager.py` | `test_exact_boundaries`, `test_session_expiry_and_restore` |
| languages, thresholds, timings configurable | ✅ | `config/default.yaml` | config + session tests |
| session persistence across restarts | 🔴 | | |
| unseen multilingual conversations | ⚠️ rule-based coverage is finite | | fuzz run only (no crashes) |

## Not yet verified (do before submitting)
* Run the app yourself and try your own conversations in each language.
* Read the hidden-evaluation notes on the intern portal for anything this matrix misses.
* Confirm `uv sync && uv run pytest` passes on **your** machine.
