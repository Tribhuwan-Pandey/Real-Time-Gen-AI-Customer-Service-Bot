"""Knowledge-base update pipeline (Task 3).

ingest  -> only NEW or MODIFIED files are parsed; duplicates skipped; invalid files quarantined
quality -> candidate must not be worse than the live KB (accuracy, grounding) and must meet minimums
activate-> only inside the maintenance window; otherwise held as 'approved_pending_window'
health  -> checks for `kb_pipeline.health_check_minutes` after activation; failure => automatic rollback
access  -> update / rollback require permissions (AuthorizationService)
Documents are untrusted data: instruction-like sentences are stripped and PII masked while parsing.
"""
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

from ..core.auth import AuthorizationService, PermissionDenied
from ..core.clock import Clock
from ..core.config import Config
from ..core.logging import get_logger
from ..monitoring.metrics import MetricsCollector
from ..rag.answerer import KnowledgeAssistant
from ..rag.loader import ALLOWED_SUFFIXES, InvalidDocument, parse_document
from ..rag.models import Document
from ..rag.retriever import KnowledgeIndex, Retriever
from .store import KBVersion, RollbackError, VersionedStore

log = get_logger("kb_pipeline")
MAX_FILE_BYTES = 5 * 1024 * 1024


@dataclass
class IngestReport:
    documents: dict[str, Document] = field(default_factory=dict)
    manifest: dict[str, str] = field(default_factory=dict)
    new: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    duplicates: dict[str, str] = field(default_factory=dict)       # file -> duplicate of
    quarantined: dict[str, str] = field(default_factory=dict)      # file -> reason
    removed: list[str] = field(default_factory=list)
    injection_flagged: dict[str, list[str]] = field(default_factory=dict)

    @property
    def changed(self) -> bool:
        return bool(self.new or self.modified or self.removed)


@dataclass
class QualityReport:
    accuracy: float
    grounding: float
    cases: int
    failures: list[str]
    baseline_accuracy: float | None = None
    baseline_grounding: float | None = None
    passed: bool = True
    reasons: list[str] = field(default_factory=list)


@dataclass
class UpdateResult:
    status: str            # no_changes | rejected | activated | approved_pending_window
    ingest: IngestReport | None = None
    quality: QualityReport | None = None
    version_id: int | None = None
    detail: str = ""


@dataclass
class HealthResult:
    status: str            # healthy | rolled_back | unhealthy_no_rollback | confirmed | idle
    detail: str = ""


@dataclass
class _Watch:
    version_id: int
    activated_at: datetime
    deadline: datetime
    checked: bool = False


class KBPipeline:
    def __init__(self, config: Config, clock: Clock, auth: AuthorizationService, store: VersionedStore,
                 docs_dir: Path, quarantine_dir: Path, eval_cases: list[dict],
                 metrics: MetricsCollector | None = None, health_check: Callable[[], bool] | None = None):
        self._config, self._clock, self._auth, self.store = config, clock, auth, store
        self.docs_dir, self.quarantine_dir = Path(docs_dir), Path(quarantine_dir)
        self.eval_cases = eval_cases
        self.metrics = metrics
        self.health_check = health_check or self._default_health_check
        self.pending: tuple[KBVersion, IngestReport] | None = None
        self.watch: _Watch | None = None
        self.parse_calls = 0
        self._rejected_fingerprint: str | None = None

    def _authorize(self, role: str | None, permission: str) -> None:
        """Permission check that leaves an audit trail (log + metric) when access is denied."""
        try:
            self._auth.require(role, permission)
        except PermissionDenied:
            log.warning("unauthorised '%s' attempt by role=%s", permission, role)
            if self.metrics:
                self.metrics.record_failure("unauthorized_access")
            raise

    # ---------------------------------------------------------------- time helpers
    def _local(self, dt: datetime) -> datetime:
        return dt.astimezone(ZoneInfo(self._config.get("business.timezone")))

    def in_maintenance_window(self, now: datetime | None = None) -> bool:
        local = self._local(now or self._clock.now()).time().replace(tzinfo=None)
        start = datetime.strptime(self._config.get("kb_pipeline.maintenance_window.start"), "%H:%M").time()
        end = datetime.strptime(self._config.get("kb_pipeline.maintenance_window.end"), "%H:%M").time()
        if start <= end:
            return start <= local < end
        return local >= start or local < end            # window crossing midnight

    # ---------------------------------------------------------------- ingest
    def _quarantine(self, path: Path, reason: str, report: IngestReport) -> None:
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(self.quarantine_dir / path.name))
        (self.quarantine_dir / f"{path.name}.reason.txt").write_text(reason, encoding="utf-8")
        report.quarantined[path.name] = reason
        log.warning("quarantined %s: %s", path.name, reason)
        if self.metrics:
            self.metrics.record_failure("quarantined_document")

    def ingest(self) -> IngestReport:
        if not self.docs_dir.is_dir():
            raise OSError(f"knowledge source unavailable: {self.docs_dir}")
        active = self.store.active
        report = IngestReport(documents=dict(active.documents) if active else {},
                              manifest=dict(active.manifest) if active else {})
        known_levels = set(self._config.get("auth.access_levels"))
        hash_owner = {h: n for n, h in report.manifest.items()}
        seen_files: set[str] = set()
        for path in sorted(self.docs_dir.iterdir()):
            if not path.is_file() or path.name.startswith("."):
                continue
            seen_files.add(path.name)
            if path.suffix.lower() not in ALLOWED_SUFFIXES:
                self._quarantine(path, f"unsupported file type '{path.suffix}'", report)
                continue
            data = path.read_bytes()
            if len(data) > MAX_FILE_BYTES:
                self._quarantine(path, "file too large", report)
                continue
            digest = hashlib.sha256(data).hexdigest()
            if report.manifest.get(path.name) == digest:
                report.unchanged.append(path.name)               # not re-processed
                continue
            owner = hash_owner.get(digest)
            if owner and owner != path.name and (self.docs_dir / owner).exists():
                report.duplicates[path.name] = owner             # same bytes as another LIVE file
                continue                                         # (a rename has no live owner -> processed)
            self.parse_calls += 1
            try:
                doc = parse_document(data, str(path), known_levels=known_levels)
                for other_name, other in report.documents.items():
                    if (other_name != path.name and other.doc_id == doc.doc_id and other.version == doc.version
                            and other.content_hash != doc.content_hash):
                        raise InvalidDocument(f"conflicts with {other_name}: same doc_id/version, different content")
            except InvalidDocument as exc:
                self._quarantine(path, str(exc), report)
                continue
            (report.modified if path.name in report.documents else report.new).append(path.name)
            report.documents[path.name] = doc
            report.manifest[path.name] = digest
            hash_owner[digest] = path.name
            if doc.sanitize_findings:
                report.injection_flagged[path.name] = list(doc.sanitize_findings)
                if self.metrics:
                    self.metrics.record_failure("prompt_injection_in_document")
        for name in list(report.documents):
            if name not in seen_files and name not in report.quarantined:
                del report.documents[name]
                report.manifest.pop(name, None)
                report.removed.append(name)
        return report

    # ---------------------------------------------------------------- quality gate
    def _assistant_for(self, documents: list[Document]) -> KnowledgeAssistant:
        index = KnowledgeIndex(documents)
        return KnowledgeAssistant(Retriever(lambda: index, self._auth, self._config), self._config, self._clock)

    def _score(self, documents: list[Document]) -> tuple[float, float, list[str]]:
        assistant = self._assistant_for(documents)
        correct, grounds, failures = 0, [], []
        for case in self.eval_cases:
            ans = assistant.answer(case["question"], case.get("role", "customer"), case.get("region"))
            if case.get("expect_refusal"):
                ok = ans.status == "refused"
            else:
                ok = (ans.status == "answered" and bool(ans.citations)
                      and ans.citations[0]["doc_id"] == case["expect_doc"]
                      and case["expect_contains"].lower() in ans.text.lower())
            if ans.status == "answered":
                grounds.append(ans.grounding)
            if ok:
                correct += 1
            else:
                failures.append(case["question"])
        n = len(self.eval_cases) or 1
        return round(correct / n, 3), round(sum(grounds) / len(grounds), 3) if grounds else 1.0, failures

    def evaluate(self, candidate: dict[str, Document]) -> QualityReport:
        acc, grd, failures = self._score(list(candidate.values()))
        active = self.store.active
        base_acc = base_grd = None
        if active:
            base_acc, base_grd, _ = self._score(list(active.documents.values()))
        c = self._config
        reasons = []
        if acc < c.get("kb_pipeline.min_accuracy"):
            reasons.append(f"accuracy {acc} below minimum {c.get('kb_pipeline.min_accuracy')}")
        if grd < c.get("kb_pipeline.min_grounding"):
            reasons.append(f"grounding {grd} below minimum {c.get('kb_pipeline.min_grounding')}")
        if base_acc is not None and acc < base_acc - c.get("kb_pipeline.max_accuracy_drop"):
            reasons.append(f"accuracy dropped from {base_acc} to {acc}")
        if base_grd is not None and grd < base_grd - c.get("kb_pipeline.max_grounding_drop"):
            reasons.append(f"grounding dropped from {base_grd} to {grd}")
        return QualityReport(acc, grd, len(self.eval_cases), failures, base_acc, base_grd,
                             passed=not reasons, reasons=reasons)

    # ---------------------------------------------------------------- update / activation
    def run_update(self, role: str | None) -> UpdateResult:
        self._authorize(role, "kb_update")               # raises PermissionDenied (and is audited)
        report = self.ingest()
        if not report.changed:
            return UpdateResult("no_changes", report, detail="no new or modified documents")
        # the verdict depends on the files, the thresholds, the eval cases and the live version: hash all of them
        verdict_inputs = {"manifest": report.manifest, "eval": self.eval_cases,
                          "thresholds": {k: self._config.get(f"kb_pipeline.{k}") for k in
                                         ("min_accuracy", "min_grounding", "max_accuracy_drop", "max_grounding_drop")},
                          "rag": {k: self._config.get(f"rag.{k}") for k in ("min_score", "top_k", "min_support_ratio")},
                          "active": self.store.active_id}
        fingerprint = hashlib.sha256(json.dumps(verdict_inputs, sort_keys=True, default=str).encode()).hexdigest()
        if fingerprint == self._rejected_fingerprint:
            return UpdateResult("rejected", report, detail="identical to a previously rejected candidate")
        quality = self.evaluate(report.documents)
        if not quality.passed:
            self._rejected_fingerprint = fingerprint
            log.warning("update rejected: %s", "; ".join(quality.reasons))
            if self.metrics:
                self.metrics.record_failure("update_rejected")
            return UpdateResult("rejected", report, quality, detail="; ".join(quality.reasons))
        self._rejected_fingerprint = None
        version = self.store.add_version(report.documents, report.manifest, note="approved")
        if self.in_maintenance_window():
            self._activate(version)
            return UpdateResult("activated", report, quality, version.version_id)
        self.pending = (version, report)
        return UpdateResult("approved_pending_window", report, quality, version.version_id,
                            "waiting for the maintenance window")

    def bootstrap(self, role: str | None) -> UpdateResult:
        """First load of the knowledge base (nothing live yet, so there is nothing to disrupt: the
        maintenance window does not apply). The quality report is still produced and stored."""
        self._authorize(role, "kb_update")
        if self.store.active is not None:
            raise RuntimeError("knowledge base already loaded; use run_update()")
        report = self.ingest()
        quality = self.evaluate(report.documents)
        version = self.store.add_version(report.documents, report.manifest, note="bootstrap")
        self._activate(version)
        return UpdateResult("activated", report, quality, version.version_id, "initial load")

    def _activate(self, version: KBVersion) -> None:
        now = self._clock.now()
        self.store.activate(version.version_id)
        self.pending = None
        self.watch = _Watch(version.version_id, now,
                            now + timedelta(minutes=self._config.get("kb_pipeline.health_check_minutes")))
        log.info("activated knowledge base version %s", version.version_id)

    def activate_pending(self) -> bool:
        """Activate an approved update once the maintenance window opens."""
        if self.pending and self.in_maintenance_window():
            self._activate(self.pending[0])
            return True
        return False

    # ---------------------------------------------------------------- health & rollback
    def _default_health_check(self) -> bool:
        return self.smoke_health_check()

    def smoke_health_check(self, sample_size: int = 5) -> bool:
        """Is the LIVE knowledge base actually usable? It must contain documents and answer sample
        questions without errors (an exception = outage) with at least one real answer."""
        active = self.store.active
        if not active or not active.documents:
            return False
        sample = [c for c in self.eval_cases if not c.get("expect_refusal")][:sample_size]
        if not sample:
            return True
        assistant = self._assistant_for(list(active.documents.values()))
        answered = 0
        for case in sample:
            a = assistant.answer(case["question"], case.get("role", "customer"), case.get("region"))
            answered += a.status == "answered"
        return answered >= 1

    def health_tick(self) -> HealthResult:
        w = self.watch
        if w is None:
            return HealthResult("idle")
        now = self._clock.now()
        try:
            ok = bool(self.health_check())
            error = "" if ok else "health check returned unhealthy"
        except Exception as exc:                          # service outage
            ok, error = False, f"health check error: {type(exc).__name__}"
        w.checked = True
        if not ok:
            if self._auto_rollback(error):
                return HealthResult("rolled_back", error)
            return HealthResult("unhealthy_no_rollback", error + "; no previous version available")
        if now >= w.deadline:
            self.watch = None
            return HealthResult("confirmed", "healthy for the whole observation period")
        return HealthResult("healthy")

    def _auto_rollback(self, reason: str) -> bool:
        log.error("automatic rollback: %s", reason)
        if self.metrics:
            self.metrics.record_failure("auto_rollback")
        self.watch = None
        try:
            self.store.rollback()
            return True
        except RollbackError:
            return False

    def rollback(self, role: str | None) -> KBVersion:
        self._authorize(role, "kb_rollback")
        return self.store.rollback()
