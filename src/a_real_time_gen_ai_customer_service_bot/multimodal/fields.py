"""Extract order IDs, dates, amounts, products and error codes from OCR text; compare with the customer's message.

Values come back exactly as read. If a value is not present it is simply absent - never guessed.
"""
import re
from dataclasses import dataclass, field

from ..core.config import Config
from ..core.masking import mask_text
from ..core.text import tokens
from ..language.entities import extract_entities, parse_date

_PRODUCT_LINE = re.compile(r"(?:product|item|description)\s*[:\-]\s*([^\n]{2,60})", re.I)


@dataclass
class Conflict:
    field: str
    message_value: str
    document_value: str


@dataclass
class DocumentFields:
    order_ids: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    amounts: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any((self.order_ids, self.dates, self.amounts, self.products, self.error_codes))

    def summary(self) -> str:
        parts = []
        for label, values in (("order", self.order_ids), ("date", self.dates), ("amount", self.amounts),
                              ("product", self.products), ("error", self.error_codes)):
            if values:
                parts.append(f"{label}: {', '.join(values)}")
        return mask_text("; ".join(parts)) or "no readable fields"

    def to_dict(self) -> dict:
        return {"order_ids": self.order_ids, "dates": self.dates, "amounts": self.amounts,
                "products": self.products, "error_codes": self.error_codes}


def extract_document_fields(text: str, config: Config) -> DocumentFields:
    e = extract_entities(text)
    products = list(e["product_codes"])
    products += [m.group(1).strip() for m in _PRODUCT_LINE.finditer(text)]
    known = {p.lower() for p in config.get("tickets.known_products")}
    products += [t for t in dict.fromkeys(tokens(text)) if t in known and t not in {p.lower() for p in products}]
    return DocumentFields(e["order_ids"], e["dates"], e["amounts"], list(dict.fromkeys(products)), e["error_codes"])


def amount_value(raw: str) -> float | None:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", raw)
    return float(m.group(0).replace(",", "")) if m else None


def _norm(kind: str, value: str):
    if kind == "amounts":
        return amount_value(value)
    if kind == "dates":
        return parse_date(value) or value.lower()
    return value.lower().replace(" ", "").replace("_", "-")


def compare_with_message(doc: DocumentFields, message: dict[str, list[str]]) -> list[Conflict]:
    """A conflict exists when BOTH sides state a value for a field and none of them agree."""
    pairs = [("order_ids", doc.order_ids, message.get("order_ids", [])),
             ("amounts", doc.amounts, message.get("amounts", [])),
             ("dates", doc.dates, message.get("dates", [])),
             ("products", doc.products, message.get("product_codes", [])),
             ("error_codes", doc.error_codes, message.get("error_codes", []))]
    conflicts = []
    for kind, doc_values, msg_values in pairs:
        if not doc_values or not msg_values:
            continue
        d = {_norm(kind, v) for v in doc_values}
        m = {_norm(kind, v) for v in msg_values}
        if not d & m:
            conflicts.append(Conflict(kind, msg_values[-1], doc_values[0]))
    return conflicts
