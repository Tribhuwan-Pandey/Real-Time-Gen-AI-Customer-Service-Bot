"""Multimodal analysis (Task 5).

validate -> store -> OCR (30 s budget, then background queue) -> quality check -> hidden-instruction check
         -> extract fields -> compare with the customer's message / conversation -> result

Rules enforced here:
  * unsafe files and files carrying hidden instructions are REJECTED (and deleted)
  * unreadable / blurred input -> ask for another file; nothing is guessed
  * conflicting evidence -> ask the customer to clarify
  * logs contain only masked summaries, never raw OCR text
"""
import itertools
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field

from ..core.config import Config
from ..core.logging import get_logger
from ..core.masking import mask_text
from ..core.security import detect_injection
from .fields import Conflict, DocumentFields, compare_with_message, extract_document_fields
from .ocr import OCREngine, OCRUnavailable
from .storage import FileStore
from .validator import validate_file

log = get_logger("multimodal")


@dataclass
class AttachmentResult:
    status: str     # processed | rejected | needs_better_file | needs_clarification | queued | failed
    reason: str = ""
    file_id: str | None = None
    filename: str = ""
    fields: DocumentFields | None = None
    conflicts: list[Conflict] = field(default_factory=list)
    summary: str = ""
    reasons: list[str] = field(default_factory=list)
    job_id: str | None = None
    customer_id: str = ""


@dataclass
class _Job:
    job_id: str
    customer_id: str
    filename: str
    file_id: str
    future: Future
    notified: bool = False


class MultimodalService:
    def __init__(self, config: Config, ocr: OCREngine, store: FileStore, executor: ThreadPoolExecutor | None = None):
        self._config, self._ocr, self._store = config, ocr, store
        self._executor = executor or ThreadPoolExecutor(max_workers=4)
        self._jobs: list[_Job] = []
        self._ids = itertools.count(1)

    # --------------------------------------------------------------------------------------
    def process(self, customer_id: str, filename: str, data: bytes,
                known_entities: dict[str, list[str]] | None = None) -> AttachmentResult:
        check = validate_file(filename, data, self._config)
        if not check.ok:
            log.warning("rejected upload from %s: %s", customer_id, check.reasons)
            return AttachmentResult("rejected", "unsafe_file", filename=_safe_name(filename),
                                    reasons=check.reasons, customer_id=customer_id)
        stored = self._store.save(customer_id, data, check.extension)
        future = self._executor.submit(self._analyze, stored.file_id, filename, data, known_entities or {}, customer_id)
        try:
            return future.result(timeout=self._config.get("multimodal.processing_timeout_seconds"))
        except FutureTimeout:
            job = _Job(f"J-{next(self._ids):04d}", customer_id, _safe_name(filename), stored.file_id, future)
            self._jobs.append(job)
            log.info("processing of %s exceeded the time budget; moved to background as %s", stored.file_id, job.job_id)
            return AttachmentResult("queued", "background_processing", file_id=stored.file_id,
                                    filename=job.filename, job_id=job.job_id, customer_id=customer_id)
        except Exception as exc:
            log.error("unexpected processing error: %s", type(exc).__name__)
            return AttachmentResult("failed", "processing_error", file_id=stored.file_id, customer_id=customer_id)

    def _analyze(self, file_id: str, filename: str, data: bytes, known: dict, customer_id: str) -> AttachmentResult:
        base = dict(file_id=file_id, filename=_safe_name(filename), customer_id=customer_id)
        try:
            ocr = self._ocr.extract(data, filename)
        except OCRUnavailable as exc:
            return AttachmentResult("failed", "ocr_unavailable", reasons=[str(exc)], **base)
        text = ocr.text or ""
        if detect_injection(text):                                   # instructions hidden in the image/document
            self._store.delete(file_id)
            log.warning("hidden instructions found in %s; file rejected and deleted", file_id)
            return AttachmentResult("rejected", "hidden_instructions", reasons=["instruction-like content in file"], **base)
        if (ocr.confidence < self._config.get("multimodal.min_ocr_confidence")
                or len(text.strip()) < self._config.get("multimodal.min_text_chars")):
            return AttachmentResult("needs_better_file", "low_quality",
                                    reasons=[f"ocr confidence {ocr.confidence:.2f}"], **base)
        fields = extract_document_fields(text, self._config)
        if fields.is_empty():
            return AttachmentResult("needs_better_file", "nothing_readable", fields=fields, **base)
        conflicts = compare_with_message(fields, known)
        summary = fields.summary()
        log.info("processed %s: %s", file_id, summary)               # masked summary only
        if conflicts:
            return AttachmentResult("needs_clarification", "conflict", fields=fields, conflicts=conflicts,
                                    summary=summary, **base)
        return AttachmentResult("processed", fields=fields, summary=summary, **base)

    # --------------------------------------------------------------------------------------
    def collect_background(self) -> list[AttachmentResult]:
        """Results of background jobs that finished since the last call (used to notify customers)."""
        out = []
        for job in self._jobs:
            if job.notified or not job.future.done():
                continue
            job.notified = True
            try:
                res = job.future.result()
            except Exception:
                res = AttachmentResult("failed", "processing_error", file_id=job.file_id, customer_id=job.customer_id)
            res.job_id = job.job_id
            out.append(res)
        return out

    def pending_jobs(self) -> int:
        return sum(1 for j in self._jobs if not j.notified)

    def purge_expired(self) -> list[str]:
        return self._store.purge_expired()


def _safe_name(filename: str) -> str:
    return mask_text((filename or "").replace("\\", "/").rsplit("/", 1)[-1])[:80]
