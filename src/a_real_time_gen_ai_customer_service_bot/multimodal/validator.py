"""Reject unsafe uploads BEFORE anything is stored or parsed."""
import re
from dataclasses import dataclass, field

from ..core.config import Config

_DANGEROUS_EXT = {"exe", "sh", "bat", "cmd", "com", "js", "jar", "msi", "vbs", "ps1", "php", "py", "dll", "scr", "html", "svg", "apk"}
_SIGNATURES = {"png": b"\x89PNG\r\n\x1a\n", "jpg": b"\xff\xd8\xff", "jpeg": b"\xff\xd8\xff", "pdf": b"%PDF-"}
_EXEC_MAGIC = (b"MZ", b"\x7fELF", b"#!", b"PK\x03\x04")
_PDF_ACTIVE = re.compile(rb"/(?:JavaScript|JS|Launch|OpenAction|EmbeddedFile|RichMedia)\b")
_TEXT_ACTIVE = re.compile(r"<\s*script|<\?php|javascript:", re.I)


@dataclass
class ValidationResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    extension: str = ""


def validate_file(filename: str, data: bytes, config: Config) -> ValidationResult:
    reasons: list[str] = []
    name = filename or ""
    if "/" in name or "\\" in name or "\x00" in name or ".." in name:
        reasons.append("unsafe file name")
    parts = name.lower().rsplit("/", 1)[-1].split(".")
    ext = parts[-1] if len(parts) > 1 else ""
    if set(parts[1:]) & _DANGEROUS_EXT:
        reasons.append("dangerous file type")
    elif ext not in config.get("multimodal.allowed_extensions"):
        reasons.append(f"file type '.{ext}' is not allowed")
    max_bytes = config.get("multimodal.max_file_mb") * 1024 * 1024
    if not data:
        reasons.append("file is empty")
    elif len(data) > max_bytes:
        reasons.append("file too large")
    if data and any(data.startswith(m) for m in _EXEC_MAGIC):
        reasons.append("file content looks executable or archived")
    if data and ext in _SIGNATURES and not data.startswith(_SIGNATURES[ext]):
        reasons.append("file content does not match its extension")
    if data and ext == "pdf" and _PDF_ACTIVE.search(data):
        reasons.append("PDF contains active content")
    if data and ext == "txt":
        try:
            text = data.decode("utf-8")
            if "\x00" in text:
                reasons.append("binary data in text file")
            elif _TEXT_ACTIVE.search(text):
                reasons.append("text file contains script content")
        except UnicodeDecodeError:
            reasons.append("text file is not valid UTF-8")
    return ValidationResult(not reasons, reasons, ext)
