"""Logging that can never leak personal or payment data: every record is masked."""
import logging as _logging

from .masking import mask_text


class MaskingFilter(_logging.Filter):
    def filter(self, record: _logging.LogRecord) -> bool:
        record.msg = mask_text(record.getMessage())
        record.args = ()
        return True


def get_logger(name: str) -> _logging.Logger:
    logger = _logging.getLogger(f"csbot.{name}")
    if not any(isinstance(f, MaskingFilter) for f in logger.filters):
        logger.addFilter(MaskingFilter())
    return logger
