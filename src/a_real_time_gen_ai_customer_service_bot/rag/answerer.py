"""RAG answering with citations, refusal, clarification and unsupported-claim detection.

Evidence handling rules:
  * every factual answer carries a [Source: ...] citation
  * missing / weak evidence -> refuse; ambiguous evidence -> ask a clarifying question
  * instructions inside retrieved documents are DATA: they are stripped and never obeyed
  * an optional generator (e.g. an LLM) may phrase the answer, but its output is verified
    sentence by sentence against the evidence; unsupported output is discarded
"""
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Callable
from zoneinfo import ZoneInfo

from ..core.clock import Clock
from ..core.config import Config
from ..core.logging import get_logger
from ..core.security import REMOVED_MARK, sanitize_untrusted
from ..core.text import fold
from ..language.entities import find_time_references
from .models import Evidence
from .query import content_tokens, query_terms, sentences, stem
from .retriever import Retriever

log = get_logger("rag.answerer")
_CITATION = re.compile(r"\[Source:[^\]]*\]")
_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")

Generator = Callable[[str, list[Evidence]], str]


@dataclass
class Answer:
    status: str                       # answered | refused | clarification
    text: str
    reason: str = ""
    citations: list[dict] = field(default_factory=list)
    grounding: float = 1.0
    as_of: date | None = None
    mode: str = "current"             # current | historical
    clarification_options: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    security_findings: list[str] = field(default_factory=list)


def citation_tag(ev: Evidence) -> str:
    d = ev.chunk.doc
    return f'[Source: {d.doc_id} v{d.version}, "{d.title}", effective {d.effective_date.isoformat()}]'


def verify_claims(text: str, evidence: list[Evidence], min_support: float) -> tuple[float, list[str]]:
    """Return (grounding ratio, unsupported sentences). A sentence is supported when enough of its
    content words appear in the evidence AND every number it states appears in the evidence."""
    ev_text = " ".join(e.chunk.text for e in evidence)
    ev_tokens = {stem(t) for t in content_tokens(ev_text)}
    ev_numbers = set(_NUMBER.findall(ev_text))
    claims = [s for s in sentences(_CITATION.sub("", text)) if content_tokens(s)]
    if not claims:
        return 1.0, []
    bad = []
    for s in claims:
        toks = {stem(t) for t in content_tokens(s)}
        ratio = len(toks & ev_tokens) / len(toks)
        numbers_ok = all(n in ev_numbers for n in _NUMBER.findall(s))
        if ratio < min_support or not numbers_ok:
            bad.append(s)
    return round(1 - len(bad) / len(claims), 3), bad


class KnowledgeAssistant:
    def __init__(self, retriever: Retriever, config: Config, clock: Clock, generator: Generator | None = None):
        self._retriever = retriever
        self._config = config
        self._clock = clock
        self._generator = generator

    def _today(self) -> date:
        return self._clock.now().astimezone(ZoneInfo(self._config.get("business.timezone"))).date()

    def answer(self, question: str, role: str | None, region: str | None = None) -> Answer:
        today = self._today()
        clean_q, findings = sanitize_untrusted(question or "")
        if findings:
            log.warning("prompt-injection patterns in question: %s", findings)
        clean_q = clean_q.replace(REMOVED_MARK, " ").strip()   # markers must not become search terms
        if not query_terms(clean_q):
            return Answer("refused", "I can't help with that request.", "no_usable_question",
                          security_findings=findings)

        # date handling: current vs historical question (explicit date, relative date, or whole month)
        refs = find_time_references(clean_q, today)
        as_of, mode, month_end = today, "current", None
        if refs:
            starts = {r.start for r in refs}
            if len(starts) > 1:
                return Answer("clarification", "Which date should I check the policy for?", "multiple_dates",
                              security_findings=findings)
            ref = refs[0]
            if ref.start > today:
                return Answer("refused", "I can only answer about policies that are in effect now or were "
                              "in effect on a past date.", "future_date", as_of=ref.start, security_findings=findings)
            if ref.kind == "month":
                month_end = min(ref.end, today)
                as_of = ref.start if ref.start <= today else today
                mode = "historical" if month_end < today else "current"
            else:
                as_of, mode = ref.start, ("historical" if ref.start < today else "current")
            folded_q = fold(clean_q)
            for r in refs:                                       # the date selects the policy; it is not a search term
                folded_q = folded_q.replace(fold(r.raw), " ")
            clean_q = folded_q

        result = self._retriever.retrieve(clean_q, role, region, as_of)
        if month_end is not None and month_end != as_of and result.evidence:
            later = self._retriever.retrieve(clean_q, role, region, month_end)
            key = lambda r: (r.evidence[0].chunk.doc.doc_id, r.evidence[0].chunk.doc.version) if r.evidence else None
            if key(later) != key(result):                        # the policy changed inside that month
                return Answer("clarification", "The policy changed during that month. Which exact date do you mean?",
                              "date_ambiguous", as_of=as_of, mode=mode, security_findings=findings)
        if result.evidence and result.evidence[0].chunk.doc.policy_key in result.needs_region:
            return Answer("clarification", "This policy differs by region. Which region applies to you?",
                          "region_ambiguous", clarification_options=result.needs_region[result.evidence[0].chunk.doc.policy_key], as_of=as_of,
                          mode=mode, security_findings=findings)
        if not result.evidence:
            return Answer("refused", "I couldn't find this in the knowledge base, so I can't answer it reliably.",
                          "no_evidence", as_of=as_of, mode=mode, security_findings=findings)

        # ambiguity: two different policies match almost equally well
        best = result.evidence[0]
        margin = self._config.get("rag.clarification_margin")
        rivals = [e for e in result.evidence[1:]
                  if e.chunk.doc.policy_key != best.chunk.doc.policy_key and e.score >= margin * best.score]
        if rivals:
            options = [best.chunk.doc.title] + [r.chunk.doc.title for r in rivals]
            return Answer("clarification", "Your question could match more than one topic. Which do you mean?",
                          "ambiguous_evidence", clarification_options=list(dict.fromkeys(options)),
                          as_of=as_of, mode=mode, security_findings=findings)

        evidence = [e for e in result.evidence if e.chunk.doc.policy_key == best.chunk.doc.policy_key]
        text = self._extractive(clean_q, evidence)
        grounding, bad = verify_claims(text, evidence, self._config.get("rag.min_support_ratio"))
        generator_rejected: list[str] = []
        if self._generator is not None:
            try:
                candidate = self._generator(clean_q, evidence)
                g2, bad2 = verify_claims(candidate, evidence, self._config.get("rag.min_support_ratio"))
            except Exception as exc:                             # LLM outage / timeout: fall back to the safe answer
                log.error("generator failed: %s", type(exc).__name__)
                candidate, g2, bad2 = None, 0.0, ["generator error"]
            if bad2 or candidate is None or not _CITATION.search(candidate):
                generator_rejected = bad2 or ["missing citation"]
                log.warning("generator output rejected: %d unsupported claim(s)", len(generator_rejected))
            else:
                text, grounding, bad = candidate, g2, bad2
        if bad:
            return Answer("refused", "I couldn't verify an answer against the knowledge base.",
                          "unsupported_claims", unsupported_claims=bad, as_of=as_of, mode=mode,
                          security_findings=findings)
        cites = [{"doc_id": e.chunk.doc.doc_id, "version": e.chunk.doc.version, "title": e.chunk.doc.title,
                  "effective_date": e.chunk.doc.effective_date.isoformat(), "region": e.chunk.doc.region}
                 for e in evidence[:1]]
        if mode == "historical":
            text = f"As of {as_of.isoformat()}: {text}"
        return Answer("answered", text, citations=cites, grounding=grounding, as_of=as_of, mode=mode,
                      unsupported_claims=generator_rejected, security_findings=findings)

    def _extractive(self, question: str, evidence: list[Evidence]) -> str:
        q = {stem(t) for t in query_terms(question)}
        best = evidence[0]
        clean, _ = sanitize_untrusted(best.chunk.text)       # defence in depth: documents are data
        sents = [s for s in sentences(clean) if REMOVED_MARK not in s]
        scored = [(len(q & {stem(t) for t in content_tokens(s)}), i, s) for i, s in enumerate(sents)]
        chosen = sorted(sorted(scored, key=lambda x: (-x[0], x[1]))[:2], key=lambda x: x[1])
        body = " ".join(s for _, _, s in chosen)
        return f"{body} {citation_tag(best)}"
