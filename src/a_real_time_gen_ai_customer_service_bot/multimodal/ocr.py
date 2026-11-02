"""OCR adapters. The service depends only on the tiny OCREngine protocol, so engines are swappable and testable.

PlainTextOCR   - for .txt 'documents' and tests (always available)
TesseractOCR   - screenshots / photos (needs `pip install pillow pytesseract` and the tesseract binary)
PdfTextOCR     - text PDFs (needs `pip install pypdf`); scanned PDFs yield low confidence -> customer is asked for a clearer file
"""
import io
from dataclasses import dataclass
from typing import Protocol


class OCRUnavailable(Exception):
    """The OCR library / binary needed for this file type is not installed."""


@dataclass
class OCRResult:
    text: str
    confidence: float          # 0..1


class OCREngine(Protocol):
    def extract(self, data: bytes, filename: str) -> OCRResult: ...


class PlainTextOCR:
    def extract(self, data: bytes, filename: str) -> OCRResult:
        return OCRResult(data.decode("utf-8", errors="replace"), 1.0)


class TesseractOCR:
    def extract(self, data: bytes, filename: str) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise OCRUnavailable("pillow/pytesseract not installed") from exc
        try:
            img = Image.open(io.BytesIO(data))
            info = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        except Exception as exc:                       # tesseract binary missing, corrupt image, ...
            raise OCRUnavailable(f"image OCR failed: {type(exc).__name__}") from exc
        words = [(w, float(c)) for w, c in zip(info["text"], info["conf"]) if w.strip() and float(c) >= 0]
        if not words:
            return OCRResult("", 0.0)
        return OCRResult(" ".join(w for w, _ in words), sum(c for _, c in words) / len(words) / 100)


class PdfTextOCR:
    def extract(self, data: bytes, filename: str) -> OCRResult:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise OCRUnavailable("pypdf not installed") from exc
        try:
            text = "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
        except Exception as exc:
            raise OCRUnavailable(f"PDF read failed: {type(exc).__name__}") from exc
        return OCRResult(text, 1.0 if len(text.strip()) > 20 else 0.0)


class DefaultOCR:
    """Dispatch by file extension."""

    def __init__(self):
        self._text, self._image, self._pdf = PlainTextOCR(), TesseractOCR(), PdfTextOCR()

    def extract(self, data: bytes, filename: str) -> OCRResult:
        ext = filename.lower().rsplit(".", 1)[-1]
        if ext == "txt":
            return self._text.extract(data, filename)
        if ext == "pdf":
            return self._pdf.extract(data, filename)
        return self._image.extract(data, filename)
