"""Uploaded-file storage with a configurable retention period. Files are stored under random ids, never user names."""
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from ..core.clock import Clock
from ..core.config import Config
from ..core.logging import get_logger

log = get_logger("filestore")


class FileExpired(Exception):
    pass


class FileMissing(Exception):
    pass


@dataclass
class StoredFile:
    file_id: str
    path: Path
    uploaded_at: datetime
    customer_id: str
    size: int


class FileStore:
    def __init__(self, base_dir: Path, config: Config, clock: Clock):
        self._dir = Path(base_dir)
        self._config, self._clock = config, clock
        self._files: dict[str, StoredFile] = {}

    def save(self, customer_id: str, data: bytes, suffix: str = "") -> StoredFile:
        self._dir.mkdir(parents=True, exist_ok=True)
        fid = "F-" + secrets.token_hex(6)
        safe_suffix = "".join(c for c in suffix if c.isalnum())[:5]
        path = self._dir / (fid + (f".{safe_suffix}" if safe_suffix else ""))
        path.write_bytes(data)
        sf = StoredFile(fid, path, self._clock.now(), customer_id, len(data))
        self._files[fid] = sf
        return sf

    def _expired(self, sf: StoredFile) -> bool:
        retention = timedelta(hours=self._config.get("multimodal.file_retention_hours"))
        return self._clock.now() - sf.uploaded_at >= retention

    def read(self, file_id: str) -> bytes:
        sf = self._files.get(file_id)
        if sf is None:
            raise FileMissing(file_id)
        if self._expired(sf):
            self.delete(file_id)
            raise FileExpired(file_id)
        return sf.path.read_bytes()

    def delete(self, file_id: str) -> None:
        sf = self._files.pop(file_id, None)
        if sf and sf.path.exists():
            sf.path.unlink()
            log.info("deleted uploaded file %s", file_id)

    def purge_expired(self) -> list[str]:
        gone = [fid for fid, sf in list(self._files.items()) if self._expired(sf)]
        for fid in gone:
            self.delete(fid)
        return gone

    def exists(self, file_id: str) -> bool:
        sf = self._files.get(file_id)
        return bool(sf and sf.path.exists())

    def ids(self) -> list[str]:
        return list(self._files)
