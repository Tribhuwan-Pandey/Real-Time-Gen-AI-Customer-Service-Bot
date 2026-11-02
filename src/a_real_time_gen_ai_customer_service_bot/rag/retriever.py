"""Retrieval with ENFORCED filtering (authorization, region, validity dates) BEFORE ranking.

Order of operations (the order matters):
  1. authorization  - documents the role may not read never reach scoring, so restricted
                      content cannot leak through the LLM, the ranking, or idf statistics
  2. region         - region-specific or "global" documents only
  3. validity       - effective_date <= as_of <= expiry_date  (future & expired policies dropped)
  4. conflict       - documents sharing a policy_key: keep only the latest applicable one
  5. ranking        - weighted term coverage, then threshold
"""
import math
from dataclasses import dataclass, field
from datetime import date
from typing import Callable

from ..core.auth import AuthorizationService
from ..core.config import Config
from ..core.logging import get_logger
from .models import Chunk, Document, Evidence
from .query import content_tokens, query_terms, stem

log = get_logger("rag.retriever")


class KnowledgeIndex:
    """Immutable snapshot of documents (one knowledge-base version)."""

    def __init__(self, documents: list[Document]):
        self.documents = list(documents)

    def chunks_for(self, docs: list[Document]) -> list[Chunk]:
        out = []
        for d in docs:
            paragraphs = [p.strip() for p in d.content.split("\n\n") if p.strip()] or [d.content]
            out.extend(Chunk(d, p, i) for i, p in enumerate(paragraphs))
        return out


@dataclass
class RetrievalResult:
    evidence: list[Evidence] = field(default_factory=list)
    as_of: date | None = None
    excluded: dict[str, int] = field(default_factory=dict)      # internal audit only, never shown to customers
    needs_region: dict[str, list[str]] = field(default_factory=dict)   # policy_key -> regions to choose from
    best_score: float = 0.0


class Retriever:
    def __init__(self, index_provider: Callable[[], KnowledgeIndex], auth: AuthorizationService, config: Config):
        self._index_provider = index_provider
        self._auth = auth
        self._config = config

    def retrieve(self, query: str, role: str | None, region: str | None, as_of: date,
                 k: int | None = None) -> RetrievalResult:
        docs = self._index_provider().documents
        excluded = {"unauthorized": 0, "region": 0, "not_yet_effective": 0, "expired": 0, "superseded": 0}
        eligible: list[Document] = []
        for d in docs:
            if not self._auth.can_read(role, d.access_level):
                excluded["unauthorized"] += 1
            elif region and d.region not in (region, "global"):
                excluded["region"] += 1
            elif d.effective_date > as_of:
                excluded["not_yet_effective"] += 1
            elif d.expiry_date is not None and as_of > d.expiry_date:
                excluded["expired"] += 1
            else:
                eligible.append(d)

        # conflict resolution: latest applicable per policy_key
        groups: dict[str, list[Document]] = {}
        for d in eligible:
            groups.setdefault(d.policy_key, []).append(d)
        resolved: list[Document] = []
        result = RetrievalResult(as_of=as_of, excluded=excluded)
        for key, members in groups.items():
            regions = {m.region for m in members}
            if region is None and len(regions) > 1:
                result.needs_region[key] = sorted(regions)
                excluded["superseded"] += len(members) - 1
                # still keep a candidate so the caller can decide; resolution deferred to the region
                resolved.append(members[0])
                continue
            members.sort(key=lambda m: (m.effective_date, m.version, m.region != "global"), reverse=True)
            resolved.append(members[0])
            excluded["superseded"] += len(members) - 1

        result.evidence = self._rank(query, resolved, k or self._config.get("rag.top_k"))
        result.best_score = result.evidence[0].score if result.evidence else 0.0
        log.info("retrieve role=%s region=%s as_of=%s excluded=%s hits=%d",
                 role, region, as_of, excluded, len(result.evidence))
        return result

    def _rank(self, query: str, docs: list[Document], k: int) -> list[Evidence]:
        q = [stem(t) for t in query_terms(query)]
        if not q or not docs:
            return []
        chunks = self._index_provider().chunks_for(docs)          # idf only over AUTHORISED chunks
        chunk_terms = [{stem(t) for t in content_tokens(c.text)} for c in chunks]
        title_terms = [{stem(t) for t in content_tokens(c.doc.title + " " + c.doc.product)} for c in chunks]
        n = len(chunks)
        idf = {t: math.log(1 + n / (1 + sum(1 for ct in chunk_terms if t in ct))) for t in q}
        total = sum(idf.values()) or 1.0
        min_cov = self._config.get("rag.min_term_coverage")
        optional = {stem(p.lower()) for p in self._config.get("tickets.known_products")}   # product names give context
        scored = []
        for c, ct, tt in zip(chunks, chunk_terms, title_terms):
            s = sum(idf[t] * (1.0 if t in ct else (0.6 if t in tt else 0.0)) for t in q) / total
            matched = [t for t in q if t in ct or t in tt]
            needed = [t for t in q if t in matched or t not in optional]   # an unmatched product name is not held against it
            coverage = len(matched) / len(needed) if needed else 0.0       # plain share of question words found
            if s > 0 and coverage >= min_cov:
                scored.append(Evidence(c, round(s, 3)))
        scored.sort(key=lambda e: (-e.score, e.chunk.doc.doc_id, e.chunk.index))
        floor = self._config.get("rag.min_score")
        return [e for e in scored if e.score >= floor][:k]
