"""Basic photo quality measures (Pillow only). Used ONLY for product photos with no readable text:
a sharp, reasonably lit, large-enough image is accepted as evidence; a blurry/dark/tiny one is not.

Note: edge variance is content dependent and noise inflates it, so it is NOT used to judge documents
(documents are judged by OCR confidence). Thresholds are configurable."""
import io
from dataclasses import dataclass

from ..core.config import Config

IMAGE_EXTENSIONS = {"png", "jpg", "jpeg"}


@dataclass
class ImageQuality:
    width: int
    height: int
    sharpness: float
    brightness: float
    contrast: float

    def problems(self, config: Config) -> list[str]:
        out = []
        if min(self.width, self.height) < config.get("multimodal.min_photo_pixels"):
            out.append("image too small")
        if self.sharpness < config.get("multimodal.min_photo_sharpness"):
            out.append("image is blurry")
        if self.brightness < config.get("multimodal.min_photo_brightness"):
            out.append("image too dark")
        if self.brightness > config.get("multimodal.max_photo_brightness"):
            out.append("image overexposed")
        if self.contrast < config.get("multimodal.min_photo_contrast"):
            out.append("image has too little contrast")
        return out


def assess_image(data: bytes) -> ImageQuality | None:
    """None when Pillow is missing or the bytes are not a decodable image."""
    try:
        from PIL import Image, ImageFilter, ImageStat
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return None
    w, h = img.size
    gray = img.convert("L")
    gray.thumbnail((1024, 1024))
    lap = gray.filter(ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1, offset=128))
    stat = ImageStat.Stat(gray)
    return ImageQuality(w, h, ImageStat.Stat(lap).var[0], stat.mean[0], stat.stddev[0])
