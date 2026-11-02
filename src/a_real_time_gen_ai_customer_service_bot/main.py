"""HTTP API (FastAPI) and a small command-line chat.

Run the API:   uv run a-real-time-gen-ai-customer-service-bot            (http://127.0.0.1:8000/docs)
Run the CLI:   uv run a-real-time-gen-ai-customer-service-bot --cli

SECURITY: the caller can NOT choose its own role. The role comes from the server-side key table
(env BOT_API_KEYS='{"<key>": "agent", "<key2>": "admin"}') matched against the X-API-Key header.
Without a valid key every caller is a plain "customer". Authorisation of knowledge documents and
admin actions is then enforced by AuthorizationService, not by the client.
"""
import asyncio
import base64
import binascii
import hmac
import json
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .core.auth import PermissionDenied
from .core.logging import get_logger
from .kb_pipeline.store import RollbackError
from .orchestrator import BotReply, Orchestrator

log = get_logger("api")
MAX_UPLOAD_B64 = 15_000_000


class AttachmentIn(BaseModel):
    filename: str = Field(max_length=200)
    content_base64: str = Field(max_length=MAX_UPLOAD_B64)


class ChatRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=120)
    message: str = Field(default="", max_length=4000)
    region: str | None = Field(default=None, max_length=10)
    attachments: list[AttachmentIn] = Field(default_factory=list, max_length=5)


def load_key_table() -> dict[str, str]:
    try:
        table = json.loads(os.environ.get("BOT_API_KEYS", "{}"))
        return table if isinstance(table, dict) else {}
    except json.JSONDecodeError:
        log.error("BOT_API_KEYS is not valid JSON; all callers will be treated as customers")
        return {}


def resolve_role(api_key: str | None, key_table: dict[str, str]) -> str:
    """Constant-time comparison against every configured key (no early exit that leaks which prefix matched)."""
    role = "customer"
    candidate = (api_key or "").encode()
    for known, known_role in key_table.items():
        if hmac.compare_digest(candidate, known.encode()):
            role = known_role
    return role


def load_env_file(path: str | None = None) -> bool:
    """Read a .env file (BOT_API_KEYS, BOT_STATE_FILE, OPENAI_API_KEY ...) if python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return bool(load_dotenv(path))


def reply_to_json(r: BotReply) -> dict:
    return {
        "text": r.text, "language": r.language, "status": r.status, "session_id": r.session_id,
        "session_event": r.session_event, "intents": [{"name": n, "confidence": c} for n, c in r.intents],
        "sentiment": ({"label": r.sentiment.label, "confidence": r.sentiment.confidence, "scores": r.sentiment.scores}
                      if r.sentiment else None),
        "escalations": [{"id": e.id, "kind": e.kind, "rule": e.rule, "reason": e.reason, "queue": e.queue,
                         "timestamp": e.timestamp.isoformat(),
                         "scheduled_for": e.scheduled_for.isoformat() if e.scheduled_for else None} for e in r.escalations],
        "ticket": ({"id": r.ticket.ticket_id, "priority": r.ticket.priority, "queue": r.ticket.queue,
                    "assigned_agent": r.ticket.assigned_agent} if r.ticket else None),
        "citations": r.answer.citations if r.answer else [],
        "attachments": [{"status": a.status, "reason": a.reason, "job_id": a.job_id} for a in r.attachments],
        "latency_ms": round(r.latency_ms, 2), "notes": r.notes,
    }


def ticket_to_json(bot: Orchestrator, t) -> dict:
    from .core.masking import mask_text
    return {
        "id": t.ticket_id, "customer_id": mask_text(t.customer_id), "order_id": t.order_id, "product": t.product,
        "issue": mask_text(t.issue), "issue_type": t.issue_type, "priority": t.priority,
        "priority_score": t.priority_score, "priority_breakdown": t.priority_breakdown, "status": t.status,
        "assigned_agent": t.assigned_agent, "queue": t.queue, "group_id": t.group_id,
        "duplicate_count": t.duplicate_count, "created_at": t.created_at.isoformat(),
        "sla": {"deadline": bot.sla.deadline(t).isoformat(), "status": bot.sla.status(t),
                "tier": t.sla_priority or t.priority, "warned": t.sla_warned, "breached": t.sla_breached},
    }


def create_app(orchestrator: Orchestrator | None = None, key_table: dict[str, str] | None = None,
               background_interval_seconds: float | None = 60.0, state_path: str | None = None) -> FastAPI:
    from .rag.llm import generator_from_env
    from .language.llm_risk import risk_classifier_from_env
    bot = orchestrator or Orchestrator(generator=generator_from_env(), risk_classifier=risk_classifier_from_env())
    keys = key_table if key_table is not None else load_key_table()
    state_file = state_path if state_path is not None else os.environ.get("BOT_STATE_FILE")
    if state_file:
        restored = bot.load_state(state_file)
        if restored:
            log.info("state restored: %s", restored)

    def persist():
        if state_file:
            try:
                bot.save_state(state_file)
            except Exception:
                log.error("saving state failed", exc_info=True)
                bot.metrics.record_failure("state_save_failed")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = None
        if background_interval_seconds:
            async def loop():
                while True:
                    await asyncio.sleep(background_interval_seconds)
                    try:
                        await asyncio.to_thread(bot.run_background_checks)       # never block the event loop
                        await asyncio.to_thread(persist)
                    except Exception:
                        log.error("background checks failed", exc_info=True)
                        bot.metrics.record_failure("background_checks")
            task = asyncio.create_task(loop())
        yield
        if task:
            task.cancel()
        persist()                                           # final save on shutdown
        bot.close()

    app = FastAPI(title="Customer Service Bot", lifespan=lifespan)
    app.state.bot = bot

    @app.get("/health")
    def health():
        active = bot.kb_store.active
        return {"status": "ok", "kb_version": active.version_id if active else None,
                "alerts": [a.name for a in bot.metrics.alerts()]}

    @app.post("/chat")
    def chat(req: ChatRequest, x_api_key: str | None = Header(default=None)):
        files = []
        for a in req.attachments:
            try:
                files.append((a.filename, base64.b64decode(a.content_base64, validate=True)))
            except (binascii.Error, ValueError):
                raise HTTPException(400, f"attachment '{a.filename}' is not valid base64") from None
        reply = bot.handle_message(req.customer_id, req.message, resolve_role(x_api_key, keys), req.region, files)
        return reply_to_json(reply)

    @app.get("/notifications/{customer_id}")
    def notifications(customer_id: str):
        return {"notifications": bot.pop_notifications(customer_id)}

    @app.get("/metrics")
    def metrics(x_api_key: str | None = Header(default=None)):
        role = resolve_role(x_api_key, keys)
        if not bot.auth.has_permission(role, "view_escalations"):
            raise HTTPException(403, "not allowed")
        return {"metrics": bot.metrics.snapshot(), "alerts": [a.__dict__ for a in bot.metrics.alerts()],
                "escalations": [{"id": e.id, "rule": e.rule, "reason": e.reason, "queue": e.queue,
                                 "timestamp": e.timestamp.isoformat(), "summary": e.summary} for e in bot.escalation.records]}

    def _guard(role: str, fn, *a):
        try:
            return fn(role, *a)
        except PermissionDenied:
            raise HTTPException(403, "not allowed") from None

    @app.get("/tickets")
    def tickets(status: str | None = None, x_api_key: str | None = Header(default=None)):
        found = _guard(resolve_role(x_api_key, keys), bot.list_tickets, status)
        return {"tickets": [ticket_to_json(bot, t) for t in found]}

    @app.get("/tickets/{ticket_id}")
    def ticket(ticket_id: str, x_api_key: str | None = Header(default=None)):
        t = _guard(resolve_role(x_api_key, keys), bot.get_ticket, ticket_id)
        if t is None:
            raise HTTPException(404, "ticket not found")
        return ticket_to_json(bot, t)

    @app.get("/tickets/{ticket_id}/handoff")
    def handoff(ticket_id: str, x_api_key: str | None = Header(default=None)):
        text = _guard(resolve_role(x_api_key, keys), bot.ticket_handoff, ticket_id)
        if text is None:
            raise HTTPException(404, "ticket not found")
        return {"ticket_id": ticket_id, "handoff_summary": text}

    @app.post("/tickets/{ticket_id}/resolve")
    def resolve(ticket_id: str, x_api_key: str | None = Header(default=None)):
        if not _guard(resolve_role(x_api_key, keys), bot.resolve_ticket, ticket_id):
            raise HTTPException(404, "ticket not found")
        return {"ticket_id": ticket_id, "status": "resolved"}

    @app.post("/admin/kb/update")
    def kb_update(x_api_key: str | None = Header(default=None)):
        try:
            r = bot.update_knowledge_base(resolve_role(x_api_key, keys))
        except PermissionDenied:
            raise HTTPException(403, "not allowed") from None
        return {"status": r.status, "detail": r.detail, "version": r.version_id}

    @app.post("/admin/kb/rollback")
    def kb_rollback(x_api_key: str | None = Header(default=None)):
        try:
            v = bot.rollback_knowledge_base(resolve_role(x_api_key, keys))
        except PermissionDenied:
            raise HTTPException(403, "not allowed") from None
        except RollbackError as exc:
            raise HTTPException(409, str(exc)) from None
        return {"active_version": v.version_id}

    return app


def run_cli() -> None:
    load_env_file()
    bot = Orchestrator()
    print("Customer service bot (type 'quit' to exit). Role: customer. Prefix a message with 'agent:' or 'admin:' to simulate a role in this local demo.")
    customer = input("Customer id: ").strip() or "cli-customer"
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ("quit", "exit"):
            break
        role = "customer"
        for r in ("agent", "admin"):
            if line.lower().startswith(f"{r}:"):
                role, line = r, line[len(r) + 1:].strip()
        reply = bot.handle_message(customer, line, role=role)
        print(f"bot> {reply.text}\n     [{reply.status} | {reply.language} | sentiment={reply.sentiment.label if reply.sentiment else '-'}]")
        bot.run_background_checks()


def main() -> None:
    if "--cli" in sys.argv:
        run_cli()
        return
    import uvicorn
    load_env_file()
    uvicorn.run(create_app(), host=os.environ.get("BOT_HOST", "127.0.0.1"), port=int(os.environ.get("BOT_PORT", "8000")))


if __name__ == "__main__":
    main()
