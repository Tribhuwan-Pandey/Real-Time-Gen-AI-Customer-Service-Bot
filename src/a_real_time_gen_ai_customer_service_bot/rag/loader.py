"""Load knowledge-base documents: Markdown/text files with a YAML front-matter metadata block.

---
doc_id: RET-POL
title: Return Policy
doc_type: policy
version: 2
product: all
region: global
access_level: public
effective_date: 2026-06-01
expiry_date: 2026-12-31      # optional
policy_key: return-policy    # optional (defaults to doc_id)
---
Body text...
"""
import hashlib
from datetime import date
from pathlib import Path

import yaml

from ..core.masking import mask_text
from ..core.security import sanitize_untrusted
from .models import Document

REQUIRED = ("doc_id", "title", "doc_type", "version", "product", "region", "access_level", "effective_date")
DOC_TYPES = {"policy", "faq", "product", "troubleshooting"}
ALLOWED_SUFFIXES = {".md", ".txt"}


class InvalidDocument(Exception):
    """The file cannot be used as a knowledge-base document (it will be quarantined)."""


def _as_date(value, name: str) -> date:
    if isinstance(value, date):
        return value
    raise InvalidDocument(f"{name} must be a YYYY-MM-DD date")


def parse_document(raw: bytes | str, source_path: str = "", sanitize: bool = True,
                   known_levels: set[str] | None = None) -> Document:
    if isinstance(raw, bytes):
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            raise InvalidDocument("file is not valid UTF-8 text") from None
        digest = hashlib.sha256(raw).hexdigest()
    else:
        text, digest = raw, hashlib.sha256(raw.encode()).hexdigest()
    if "\x00" in text:
        raise InvalidDocument("binary content")
    if not text.startswith("---"):
        raise InvalidDocument("missing metadata front matter")
    parts = text.split("---", 2)
    if len(parts) < 3:
        raise InvalidDocument("unterminated metadata front matter")
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        raise InvalidDocument("metadata is not valid YAML") from None
    if not isinstance(meta, dict):
        raise InvalidDocument("metadata must be a mapping")
    missing = [k for k in REQUIRED if meta.get(k) in (None, "")]
    if missing:
        raise InvalidDocument(f"missing metadata: {', '.join(missing)}")
    body = parts[2].strip()
    if not body:
        raise InvalidDocument("document body is empty")
    if meta["doc_type"] not in DOC_TYPES:
        raise InvalidDocument(f"doc_type must be one of {sorted(DOC_TYPES)}")
    if not isinstance(meta["version"], int) or isinstance(meta["version"], bool) or meta["version"] < 1:
        raise InvalidDocument("version must be a positive integer")
    effective = _as_date(meta["effective_date"], "effective_date")
    expiry = _as_date(meta["expiry_date"], "expiry_date") if meta.get("expiry_date") else None
    if expiry and expiry < effective:
        raise InvalidDocument("expiry_date is before effective_date")
    access = str(meta["access_level"])
    if known_levels is not None and access not in known_levels:
        raise InvalidDocument(f"unknown access_level '{access}'")

    findings: tuple = ()
    if sanitize:                       # documents are DATA: strip instruction-like sentences, mask PII
        body, found = sanitize_untrusted(body)
        findings = tuple(found)
    body = mask_text(body)
    return Document(
        doc_id=str(meta["doc_id"]), title=str(meta["title"]), content=body, doc_type=meta["doc_type"],
        version=meta["version"], product=str(meta["product"]), region=str(meta["region"]),
        access_level=access, effective_date=effective, expiry_date=expiry,
        policy_key=str(meta.get("policy_key") or meta["doc_id"]), source_path=source_path,
        content_hash=digest, sanitize_findings=findings)


def load_directory(path: Path, known_levels: set[str] | None = None) -> tuple[list[Document], dict[str, str]]:
    """Returns (valid documents, {filename: reason} for invalid ones)."""
    docs, errors = [], {}
    for f in sorted(Path(path).glob("*")):
        if f.suffix.lower() not in ALLOWED_SUFFIXES or not f.is_file():
            continue
        try:
            docs.append(parse_document(f.read_bytes(), str(f), known_levels=known_levels))
        except InvalidDocument as exc:
            errors[f.name] = str(exc)
    return docs, errors
