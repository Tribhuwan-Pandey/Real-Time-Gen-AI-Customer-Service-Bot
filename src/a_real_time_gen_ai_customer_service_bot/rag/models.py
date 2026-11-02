from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    content: str
    doc_type: str            # policy | faq | product | troubleshooting
    version: int
    product: str             # product name or "all"
    region: str              # e.g. "IN", "EU" or "global"
    access_level: str        # public | customer | internal | restricted
    effective_date: date
    expiry_date: date | None = None     # LAST valid day (inclusive); None = no expiry
    policy_key: str = ""     # documents sharing a key are versions/variants of the same policy
    source_path: str = ""
    content_hash: str = ""
    sanitize_findings: tuple = field(default_factory=tuple)

    def active_on(self, day: date) -> bool:
        return self.effective_date <= day and (self.expiry_date is None or day <= self.expiry_date)


@dataclass(frozen=True)
class Chunk:
    doc: Document
    text: str
    index: int


@dataclass
class Evidence:
    chunk: Chunk
    score: float
