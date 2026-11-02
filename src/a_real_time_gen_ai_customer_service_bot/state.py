"""Save / load the parts of the system that must survive a restart: returning-customer sessions, tickets,
escalation records and pending notifications.

NOT persisted (by design): the knowledge base (re-ingested from data/kb_docs on start), metrics windows, uploaded
files (the retention job owns them) and raw conversation text. The file contains customer contact details (tickets
need them), so it is written with owner-only permissions; put it on an encrypted volume in production.
"""
import json
import os
import tempfile
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path

from .core.logging import get_logger
from .escalation.engine import EscalationRecord
from .tickets.models import Ticket

log = get_logger("state")
VERSION = 1


def _enc(v):
    if isinstance(v, datetime):
        return {"__dt__": v.isoformat()}
    if is_dataclass(v):
        return {f.name: _enc(getattr(v, f.name)) for f in fields(v)}
    if isinstance(v, dict):
        return {k: _enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_enc(x) for x in v]
    return v


def _dec(v):
    if isinstance(v, dict):
        if set(v) == {"__dt__"}:
            return datetime.fromisoformat(v["__dt__"])
        return {k: _dec(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_dec(x) for x in v]
    return v


def dump_state(bot) -> dict:
    return {
        "version": VERSION,
        "sessions": bot.sessions.export_state(),
        "tickets": {"counter": bot.repo._n, "group_counter": bot.repo._g,
                    "items": [_enc(t) for t in bot.repo.tickets.values()]},
        "escalations": {"counter": bot.escalation._counter, "items": [_enc(r) for r in bot.escalation.records]},
        "notifications": bot.notifications,
    }


def apply_state(bot, data: dict) -> dict:
    if data.get("version") != VERSION:
        raise ValueError(f"unsupported state version {data.get('version')}")
    restored = {"sessions": bot.sessions.import_state(data.get("sessions", {}))}
    t = data.get("tickets", {})
    for item in t.get("items", []):
        ticket = Ticket(**_dec(item))
        bot.repo.tickets[ticket.ticket_id] = ticket
    bot.repo._n = max(bot.repo._n, int(t.get("counter", 0)))
    bot.repo._g = max(bot.repo._g, int(t.get("group_counter", 0)))
    e = data.get("escalations", {})
    bot.escalation.records.extend(EscalationRecord(**_dec(item)) for item in e.get("items", []))
    bot.escalation._counter = max(bot.escalation._counter, int(e.get("counter", 0)))
    bot.notifications.update(data.get("notifications", {}))
    restored["tickets"], restored["escalations"] = len(bot.repo.tickets), len(bot.escalation.records)
    return restored


def save_state(bot, path: Path) -> None:
    """Atomic write (temp file + rename) with owner-only permissions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(dump_state(bot), fh)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_state(bot, path: Path) -> dict | None:
    """Returns counts of restored items, or None when there is nothing usable (missing / corrupt file)."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return apply_state(bot, json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, KeyError, TypeError, OSError) as exc:             # corrupt file: start clean, never crash
        log.error("state file ignored: %s", type(exc).__name__)
        return None
