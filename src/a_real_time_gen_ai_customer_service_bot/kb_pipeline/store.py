from dataclasses import dataclass, field
from datetime import datetime

from ..core.clock import Clock
from ..rag.models import Document
from ..rag.retriever import KnowledgeIndex


class RollbackError(Exception):
    pass


@dataclass
class KBVersion:
    version_id: int
    created_at: datetime
    documents: dict[str, Document]            # key = source file name
    manifest: dict[str, str]                  # file name -> sha256
    status: str = "candidate"                 # candidate | active | retired | rolled_back
    note: str = ""
    activated_at: datetime | None = None


class VersionedStore:
    """Keeps every knowledge-base version. Exactly one is active; rollback restores the previous one."""

    def __init__(self, clock: Clock):
        self._clock = clock
        self.versions: dict[int, KBVersion] = {}
        self.active_id: int | None = None
        self._history: list[int] = []          # activation order
        self._next = 1

    def add_version(self, documents: dict[str, Document], manifest: dict[str, str], note: str = "") -> KBVersion:
        v = KBVersion(self._next, self._clock.now(), dict(documents), dict(manifest), note=note)
        self.versions[v.version_id] = v
        self._next += 1
        return v

    @property
    def active(self) -> KBVersion | None:
        return self.versions.get(self.active_id) if self.active_id is not None else None

    def activate(self, version_id: int) -> KBVersion:
        if version_id not in self.versions:
            raise RollbackError(f"unknown version {version_id}")
        if self.active:
            self.active.status = "retired"
        v = self.versions[version_id]
        v.status, v.activated_at, self.active_id = "active", self._clock.now(), version_id
        self._history.append(version_id)
        return v

    def rollback(self) -> KBVersion:
        """Re-activate the version that was active before the current one."""
        if len(self._history) < 2:
            raise RollbackError("no previous version to roll back to")
        bad = self._history.pop()
        previous = self._history.pop()
        restored = self.activate(previous)          # retires the bad version ...
        self.versions[bad].status = "rolled_back"   # ... then record WHY it is no longer active
        return restored

    def active_index(self) -> KnowledgeIndex:
        v = self.active
        return KnowledgeIndex(list(v.documents.values()) if v else [])
