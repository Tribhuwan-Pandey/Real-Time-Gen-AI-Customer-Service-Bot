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


def _entity_count(text: str) -> int:
    from ..language.entities import extract_entities
    e = extract_entities(text)
    return sum(len(e[k]) for k in ("order_ids", "dates", "amounts", "error_codes", "product_codes"))


class TesseractOCR:
    """Two passes (as-is and cleaned up: grey, 2x upscale, median filter, autocontrast, threshold); the pass that
    extracts MORE identifiers wins (ties: the untouched raw pass, because cleaning can truncate digits). Cleaning helps some photos and hurts others, so
    neither is trusted blindly. Unreadable results keep their low confidence and the customer is asked for a new file."""

    def _run(self, pytesseract, img) -> OCRResult:
        info = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        words = [(w, float(c)) for w, c in zip(info["text"], info["conf"]) if w.strip() and float(c) >= 0]
        if not words:
            return OCRResult("", 0.0)
        return OCRResult(" ".join(w for w, _ in words), sum(c for _, c in words) / len(words) / 100)

    @staticmethod
    def _clean(img):
        from PIL import ImageFilter, ImageOps
        g = img.convert("L")
        if g.width < 1400:
            g = g.resize((g.width * 2, g.height * 2))
        g = ImageOps.autocontrast(g.filter(ImageFilter.MedianFilter(3)), cutoff=2)
        return g.point(lambda v: 255 if v > 150 else 0)

    def extract(self, data: bytes, filename: str) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise OCRUnavailable("pillow/pytesseract not installed") from exc
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            raw = self._run(pytesseract, img)
            cleaned = self._run(pytesseract, self._clean(img))
        except Exception as exc:                       # tesseract binary missing, corrupt image, ...
            raise OCRUnavailable(f"image OCR failed: {type(exc).__name__}") from exc
        # the cleaned pass must find strictly MORE identifiers to win; on a tie the untouched image is trusted
        return cleaned if _entity_count(cleaned.text) > _entity_count(raw.text) else raw


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
        if len(text.strip()) > 20:
            return OCRResult(text, 1.0)
        # image-only (scanned) PDF: OCR the embedded page images
        parts: list[OCRResult] = []
        try:
            reader = PdfReader(io.BytesIO(data))
            for page in reader.pages[:5]:
                for image in page.images[:3]:
                    parts.append(TesseractOCR().extract(image.data, image.name or "page.png"))
        except OCRUnavailable:
            raise
        except Exception:
            return OCRResult(text, 0.0)
        if not parts:
            return OCRResult(text, 0.0)
        return OCRResult("\n".join(p.text for p in parts), sum(p.confidence for p in parts) / len(parts))


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
