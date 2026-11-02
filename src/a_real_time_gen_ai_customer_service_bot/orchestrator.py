"""The single entry point that connects all six modules.

handle_message(): session -> injection scrub -> language -> entities/corrections -> sentiment + intents
                  -> attachments -> escalation rules -> (knowledge answer | ticket | clarification) -> tone -> reply
run_background_checks(): 15-minute escalation sweep, SLA warnings/breaches, queue re-routing, session expiry,
                  file retention purge, scheduled KB update, background OCR notifications
Sentiment only changes the TONE PREFIX of a reply; it never changes which business answer is given.
"""
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import i18n
from .core.auth import AuthorizationService
from .core.business_calendar import BusinessCalendar
from .core.clock import Clock, SystemClock
from .core.config import PROJECT_ROOT, Config, load_config
from .core.logging import get_logger
from .core.masking import mask_text
from .core.security import REMOVED_MARK, sanitize_untrusted
from .escalation.engine import EscalationEngine, EscalationRecord
from .kb_pipeline.pipeline import KBPipeline
from .kb_pipeline.scheduler import UpdateScheduler
from .kb_pipeline.store import VersionedStore
from .language.detector import LanguageDetector, LanguageResult
from .language.entities import extract_entities, has_correction, merge_entities
from .language.intents import IntentMatch, detect_intents
from .monitoring.metrics import MetricsCollector
from .multimodal.ocr import DefaultOCR
from .multimodal.service import AttachmentResult, MultimodalService
from .multimodal.storage import FileStore
from .rag.answerer import Answer, KnowledgeAssistant
from .rag.query import query_terms, sentences
from .rag.retriever import Retriever
from .sentiment.analyzer import SentimentAnalyzer, SentimentResult
from .sentiment.tone import adapt_tone
from .session.manager import SessionManager
from .session.models import Conversation, Message
from .tickets.extractor import draft_from_conversation
from .tickets.models import Agent, Ticket, TicketRepository
from .tickets.routing import Router
from .tickets.service import TicketResult, TicketService
from .tickets.sla import SLAEngine

log = get_logger("orchestrator")
COMPLAINT_INTENTS = {"refund", "order_status", "technical_issue", "duplicate_payment", "account_compromise",
                     "legal_threat", "cancel_order", "complaint", "human_agent"}
REGION_WORDS = {"india": "IN", "indian": "IN", "bharat": "IN", "in": "IN", "europe": "EU", "eu": "EU",
                "global": "global", "worldwide": "global", "other": "EU"}
SYSTEM_ROLE = "kb_manager"


def default_agents() -> list[Agent]:
    return [Agent("agent-billing-1", "Billing 1", {"billing"}), Agent("agent-billing-2", "Billing 2", {"billing"}, on_call=True),
            Agent("agent-tech-1", "Tech 1", {"technical"}), Agent("agent-orders-1", "Orders 1", {"orders"}),
            Agent("agent-sec-1", "Security 1", {"security"}, on_call=True), Agent("agent-legal-1", "Legal 1", {"legal"}),
            Agent("agent-gen-1", "General 1", {"general"})]


@dataclass
class BotReply:
    text: str
    language: str
    status: str = "ok"                         # ok | clarification | escalated | error | unsupported_language
    session_id: str = ""
    session_event: str = ""
    intents: list[tuple[str, float]] = field(default_factory=list)
    sentiment: SentimentResult | None = None
    escalations: list[EscalationRecord] = field(default_factory=list)
    ticket: Ticket | None = None
    answer: Answer | None = None
    attachments: list[AttachmentResult] = field(default_factory=list)
    latency_ms: float = 0.0
    notes: list[str] = field(default_factory=list)


class Orchestrator:
    def __init__(self, config: Config | None = None, clock: Clock | None = None, kb_dir: Path | None = None,
                 workdir: Path | None = None, agents: list[Agent] | None = None, ocr=None, generator=None,
                 eval_cases: list[dict] | None = None, bootstrap_kb: bool = True):
        import json
        self.config = config or load_config()
        self.clock = clock or SystemClock()
        workdir = Path(workdir or PROJECT_ROOT / "data" / "runtime")
        self.calendar = BusinessCalendar(self.config)
        self.auth = AuthorizationService(self.config)
        self.metrics = MetricsCollector(self.config, self.clock)
        self.sessions = SessionManager(self.config, self.clock)
        self.detector = LanguageDetector(self.config)
        self.analyzer = SentimentAnalyzer(self.config)
        self.escalation = EscalationEngine(self.config, self.calendar, self.clock)
        self.repo = TicketRepository()
        self.sla = SLAEngine(self.config, self.calendar, self.clock, self.escalation)
        self.router = Router(self.config, self.calendar, self.clock, agents if agents is not None else default_agents(), self.repo)
        self.tickets = TicketService(self.config, self.calendar, self.clock, self.sla, self.router, self.repo)
        # knowledge base
        self.kb_store = VersionedStore(self.clock)
        cases = eval_cases if eval_cases is not None else json.loads(
            (PROJECT_ROOT / "data" / "eval" / "eval_cases.json").read_text(encoding="utf-8"))
        self.pipeline = KBPipeline(self.config, self.clock, self.auth, self.kb_store,
                                   Path(kb_dir or PROJECT_ROOT / "data" / "kb_docs"), workdir / "quarantine",
                                   cases, metrics=self.metrics)
        self.scheduler = UpdateScheduler(self.config, self.clock, self.pipeline, role=SYSTEM_ROLE)
        self.assistant = KnowledgeAssistant(
            Retriever(self.kb_store.active_index, self.auth, self.config), self.config, self.clock, generator)
        if bootstrap_kb:
            self.pipeline.bootstrap(SYSTEM_ROLE)
        # multimodal
        self.files = FileStore(workdir / "uploads", self.config, self.clock)
        self.multimodal = MultimodalService(self.config, ocr or DefaultOCR(), self.files)
        self.notifications: dict[str, list[str]] = {}

    # ------------------------------------------------------------------------------------------
    def handle_message(self, customer_id: str, text: str, role: str = "customer", region: str | None = None,
                       attachments: list[tuple[str, bytes]] | None = None) -> BotReply:
        start = time.perf_counter()
        try:
            reply = self._handle(customer_id, text or "", role, region, attachments or [])
        except Exception as exc:                                   # never crash the chat
            log.error("handler error: %s", type(exc).__name__, exc_info=True)
            self.metrics.record_failure("handler_error")
            conv = self.sessions.get_active(customer_id)
            lang = (conv.language if conv else None) or self.config.get("languages.default")
            reply = BotReply(i18n.msg(lang, "error"), lang, status="error")
        reply.latency_ms = (time.perf_counter() - start) * 1000
        self.metrics.record_latency("chat", reply.latency_ms)
        return reply

    # ------------------------------------------------------------------------------------------
    def _fmt(self, dt) -> str:
        return dt.astimezone(ZoneInfo(self.config.get("business.timezone"))).strftime("%a %d %b %H:%M")

    def _language_names(self) -> str:
        return ", ".join(i18n.LANGUAGE_NAMES.get(c, c) for c in self.config.get("languages.supported"))

    def _handle(self, customer_id, text, role, region, attachments) -> BotReply:
        cfg = self.config
        conv, event = self.sessions.get_or_create(customer_id)
        default_lang = cfg.get("languages.default")

        # 1. untrusted input: strip instruction-like sentences, keep the rest
        clean, findings = sanitize_untrusted(text)
        if findings:
            self.metrics.record_failure("prompt_injection_in_message")
            log.warning("prompt-injection patterns in customer message: %s", findings)
        clean = clean.replace(REMOVED_MARK, " ").strip()

        # 2. language (current message + session context)
        lang: LanguageResult = self.detector.detect(clean)
        confident_lang = bool(lang.language and lang.confidence >= cfg.get("confidence.language_min"))
        reply_lang = conv.language or default_lang
        if confident_lang and lang.supported:
            conv.language = reply_lang = lang.language
        notes: list[str] = []
        if confident_lang and not lang.supported:
            unsupported = BotReply(i18n.msg(conv.language or default_lang, "unsupported_language",
                                            languages=self._language_names()),
                                   conv.language or default_lang, "unsupported_language", conv.session_id, event)
            self._record(conv, text, lang.language, None, [], unsupported)
            return unsupported
        if lang.mixed:
            notes.append("mixed_language")
        if lang.transliterated:
            notes.append("transliterated")

        # 3. entities (exact values), corrections
        new_entities = extract_entities(clean)
        known_before = {k: list(v) for k, v in conv.entities.items()}
        corrected = merge_entities(conv.entities, new_entities, has_correction(clean))
        conv.corrections.extend(corrected)

        # 4. sentiment + intents
        sentiment = self.analyzer.analyze(clean, conv.sentiment_history)
        intents = detect_intents(clean)
        min_intent = cfg.get("confidence.intent_min")
        confident = [i for i in intents if i.confidence >= min_intent]
        names = [i.name for i in confident]
        pending = conv.pending

        # 5. an injection-only message gets a safe reply
        if findings and not clean:
            reply = BotReply(i18n.msg(reply_lang, "instructions_ignored"), reply_lang, "ok", conv.session_id, event,
                             notes=["prompt_injection"])
            self._record(conv, text, reply_lang, sentiment, [], reply, store_text="[removed]")
            return reply

        # 6. record the customer message (before rules that inspect history)
        conv.sentiment_history.append(sentiment)
        conv.messages.append(Message("user", clean, self.clock.now(), reply_lang, sentiment.label,
                                     sentiment.negative_like, [i.name for i in intents]))
        if names:
            conv.last_intents = names
        self.sessions.touch(conv)

        # 7. attachments
        parts: list[str] = []
        att_results = [self._attachment(conv, fn, data, known_before, new_entities, reply_lang, parts) for fn, data in attachments]

        # 8. escalation rules (risk and history, independent of tone)
        outcome = self.escalation.evaluate(conv, intents, sentiment)
        for rec in outcome.records:
            self.metrics.record_escalation(rec.rule)
        for rec in outcome.records:
            if rec.kind == "scheduled":
                parts.append(i18n.msg(reply_lang, "scheduled", ref=rec.id, when=self._fmt(rec.scheduled_for)))
            elif rec.queue == "on_call":
                parts.append(i18n.msg(reply_lang, "on_call", ref=rec.id))
            else:
                parts.append(i18n.msg(reply_lang, "escalated", ref=rec.id))

        # 9. decide the substantive answer(s)
        status, answer, ticket = "ok", None, None
        if corrected:
            parts.append(i18n.msg(reply_lang, "corrected", fields=", ".join(corrected)))
        pending_kind = pending["type"] if pending else None
        is_question = "?" in clean
        question_part = self._question_part(clean)
        wants_ticket = (bool(COMPLAINT_INTENTS & set(names)) or (sentiment.negative_like and not is_question)
                        or (pending_kind == "ticket" and not names))
        wants_kb = ("policy_question" in names or (pending_kind == "rag" and not wants_ticket)
                    or (not names and not wants_ticket and is_question and pending_kind != "ticket"
                        and bool(query_terms(question_part))))
        if wants_kb:
            question = question_part
            kb_region = region
            if pending_kind == "rag":
                question = f"{pending['question']} {clean}"
                kb_region = self._region_from(clean) or region
                conv.pending = None
            answer = self.assistant.answer(question, role, kb_region)
            parts.append(self._answer_text(answer, reply_lang, conv, question))
            if answer.status == "clarification":
                status = "clarification"
            self.metrics.record_confidence(answer.grounding if answer.status == "answered" else 0.3)
        if wants_ticket:
            result = self._ticket(conv, intents, sentiment, outcome)
            ticket = result.ticket
            parts.append(self._ticket_text(result, reply_lang, conv))
        if not parts:
            top = names[0] if names else None
            if top == "greeting":
                parts.append(i18n.msg(reply_lang, "greeting"))
            elif top in ("thanks", "resolved"):
                parts.append(i18n.msg(reply_lang, "resolved" if top == "resolved" else "thanks"))
            elif not lang.language and not conv.messages[:-1] and not names:
                parts.append(i18n.msg(default_lang, "clarify_language", languages=self._language_names()))
                status = "clarification"
            else:
                parts.append(i18n.msg(reply_lang, "clarify_intent"))
                status = "clarification"
            self.metrics.record_confidence(intents[0].confidence if intents else 0.0)
        elif not wants_kb:
            self.metrics.record_confidence(intents[0].confidence if intents else sentiment.confidence)
        if outcome.escalated:
            status = "escalated"
        if event == "restored" and len(conv.user_messages()) == 1 and conv.restored_summary:
            parts.insert(0, i18n.msg(reply_lang, "welcome_back", summary=conv.restored_summary[:200]))

        body = "\n".join(p for p in parts if p)
        text_out = body if sentiment.low_confidence else adapt_tone(body, sentiment, reply_lang)
        reply = BotReply(text_out, reply_lang, status, conv.session_id, event, [(i.name, i.confidence) for i in intents],
                         sentiment, outcome.records, ticket, answer, att_results, notes=notes)
        conv.messages.append(Message("assistant", text_out, self.clock.now(), reply_lang))
        self.sessions.touch(conv)
        return reply

    # ------------------------------------------------------------------------------------------
    def _record(self, conv, text, lang, sentiment, intents, reply, store_text=None):
        conv.messages.append(Message("user", store_text if store_text is not None else mask_text(text), self.clock.now(), lang))
        conv.messages.append(Message("assistant", reply.text, self.clock.now(), lang))
        self.sessions.touch(conv)

    @staticmethod
    def _question_part(text: str) -> str:
        """The sentences that actually ask something (so greetings and side remarks do not dilute retrieval)."""
        qs = [s for s in sentences(text) if "?" in s]
        return " ".join(qs) if qs else text

    @staticmethod
    def _region_from(text: str) -> str | None:
        for tok in text.lower().replace(",", " ").replace("?", " ").split():
            if tok in REGION_WORDS:
                return REGION_WORDS[tok]
        return None

    def _answer_text(self, a: Answer, lang: str, conv: Conversation, question: str) -> str:
        if a.status == "answered":
            return a.text
        if a.reason == "region_ambiguous":
            conv.pending = {"type": "rag", "question": question}
            return i18n.msg(lang, "ask_region", options=", ".join(a.clarification_options))
        if a.status == "clarification":
            if a.reason == "multiple_dates":
                return i18n.msg(lang, "ask_date")
            conv.pending = {"type": "rag", "question": question}
            return i18n.msg(lang, "clarify_options", question=a.text, options=" / ".join(a.clarification_options))
        if a.reason == "future_date":
            return i18n.msg(lang, "refused_future")
        if a.reason == "no_usable_question":
            return i18n.msg(lang, "clarify_intent")
        return i18n.msg(lang, "refused")

    def _ticket(self, conv, intents: list[IntentMatch], sentiment, outcome) -> TicketResult:
        high_risk = set(self.config.get("escalation.high_risk_intents"))
        seen = {i for m in conv.user_messages() for i in m.intents} | {i.name for i in intents}
        draft = draft_from_conversation(conv, intents, self.config, sorted(seen & high_risk))
        result = self.tickets.create(draft)
        if result.status == "needs_info":
            conv.pending = {"type": "ticket", "missing": result.missing}
        else:
            conv.pending = None if (conv.pending or {}).get("type") == "ticket" else conv.pending
            if result.ticket and result.ticket.ticket_id not in conv.ticket_ids:
                conv.ticket_ids.append(result.ticket.ticket_id)
        return result

    def _ticket_text(self, r: TicketResult, lang: str, conv) -> str:
        if r.status == "needs_info":
            return i18n.msg(lang, "ticket_needs", fields=i18n.field_names(lang, r.missing))
        t = r.ticket
        if r.status == "duplicate":
            return i18n.msg(lang, "ticket_duplicate", ticket=t.ticket_id)
        return i18n.msg(lang, "ticket_created", ticket=t.ticket_id, priority=t.priority,
                        deadline=self._fmt(self.sla.deadline(t)))

    def _attachment(self, conv, filename, data, known_before, new_entities, lang, parts) -> AttachmentResult:
        known = {k: list(known_before.get(k, [])) + [v for v in new_entities.get(k, []) if v not in known_before.get(k, [])]
                 for k in set(known_before) | set(new_entities)}
        res = self.multimodal.process(conv.customer_id, filename, data, known)
        if res.status == "processed":
            conv.attachments.append({"filename": res.filename, "summary": res.summary, "file_id": res.file_id})
            f = res.fields
            merge_entities(conv.entities, {"order_ids": f.order_ids if not known.get("order_ids") else [],
                                           "error_codes": f.error_codes, "amounts": f.amounts if not known.get("amounts") else [],
                                           "dates": f.dates if not known.get("dates") else []}, False)
            parts.append(i18n.msg(lang, "file_ok", summary=res.summary))
        elif res.status == "rejected":
            parts.append(i18n.msg(lang, "file_hidden" if res.reason == "hidden_instructions" else "file_rejected"))
        elif res.status == "needs_better_file":
            parts.append(i18n.msg(lang, "file_low_quality"))
        elif res.status == "needs_clarification":
            c = res.conflicts[0]
            parts.append(i18n.msg(lang, "file_conflict", field=c.field, doc_value=mask_text(c.document_value),
                                  msg_value=mask_text(c.message_value)))
        elif res.status == "queued":
            parts.append(i18n.msg(lang, "file_queued", job=res.job_id))
        else:
            parts.append(i18n.msg(lang, "file_failed"))
        return res

    # ------------------------------------------------------------------------------------------
    def run_background_checks(self) -> dict[str, Any]:
        """Everything that must happen even when nobody is chatting. Call it every minute."""
        convs = self.sessions.active_conversations()
        escalations = self.escalation.check_timeouts(convs)          # BEFORE sessions expire
        for rec in escalations:
            self.metrics.record_escalation(rec.rule)
        sla_events = self.tickets.run_sla_checks()
        for ev in sla_events:
            if ev.escalation:
                self.metrics.record_escalation("sla_breach")
        rerouted = self.router.reroute_waiting()
        ended = self.sessions.sweep()
        purged = self.files.purge_expired()
        sched = self.scheduler.tick()
        for res in self.multimodal.collect_background():
            conv = self.sessions.get_active(res.customer_id)
            lang = (conv.language if conv else None) or self.config.get("languages.default")
            text = (i18n.msg(lang, "file_done", job=res.job_id, summary=res.summary or res.status)
                    if res.status == "processed" else i18n.msg(lang, "file_low_quality" if res.status == "needs_better_file" else "file_failed"))
            self.notifications.setdefault(res.customer_id, []).append(text)
        return {"escalations": escalations, "sla_events": sla_events, "rerouted": rerouted,
                "sessions_ended": ended, "files_purged": purged, "scheduler": sched}

    def pop_notifications(self, customer_id: str) -> list[str]:
        return self.notifications.pop(customer_id, [])

    # -- knowledge-base administration (role-checked) ---------------------------------------------
    def update_knowledge_base(self, role: str | None):
        return self.pipeline.run_update(role)

    def rollback_knowledge_base(self, role: str | None):
        return self.pipeline.rollback(role)
