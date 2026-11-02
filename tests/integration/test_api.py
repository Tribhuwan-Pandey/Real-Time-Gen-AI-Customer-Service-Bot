import base64
import shutil

import pytest
from fastapi.testclient import TestClient

from a_real_time_gen_ai_customer_service_bot.core.config import PROJECT_ROOT
from a_real_time_gen_ai_customer_service_bot.main import create_app, resolve_role
from a_real_time_gen_ai_customer_service_bot.orchestrator import Orchestrator

KEYS = {"agent-key": "agent", "admin-key": "admin", "kbm-key": "kb_manager"}


@pytest.fixture
def client(tmp_path, config, clock):
    kb = tmp_path / "kb"
    shutil.copytree(PROJECT_ROOT / "data" / "kb_docs", kb)
    bot = Orchestrator(config, clock, kb_dir=kb, workdir=tmp_path / "run")
    return TestClient(create_app(bot, KEYS, background_interval_seconds=None))


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["kb_version"] == 1


def test_chat_returns_cited_answer(client):
    r = client.post("/chat", json={"customer_id": "c1", "message": "How many days to return products?", "region": "EU"})
    body = r.json()
    assert r.status_code == 200 and "30 days" in body["text"] and body["citations"][0]["doc_id"] == "RET-POL"


def test_chat_escalation_is_reported(client):
    body = client.post("/chat", json={"customer_id": "c1", "message": "I was charged twice for my order"}).json()
    assert body["status"] == "escalated" and body["escalations"][0]["rule"] == "high_risk:duplicate_payment"


def test_callers_cannot_choose_their_own_role(client):
    q = {"customer_id": "c1", "message": "What is the fraud score freezing threshold?", "region": "EU", "role": "admin"}
    assert "82" not in client.post("/chat", json=q).json()["text"]
    assert "82" not in client.post("/chat", json=q, headers={"X-API-Key": "wrong"}).json()["text"]
    assert "82" in client.post("/chat", json=q, headers={"X-API-Key": "admin-key"}).json()["text"]


def test_resolve_role():
    assert resolve_role(None, KEYS) == "customer" and resolve_role("x", KEYS) == "customer"
    assert resolve_role("admin-key", KEYS) == "admin"


def test_metrics_and_admin_endpoints_are_protected(client):
    assert client.get("/metrics").status_code == 403
    assert client.get("/metrics", headers={"X-API-Key": "agent-key"}).status_code == 200
    assert client.post("/admin/kb/update").status_code == 403
    assert client.post("/admin/kb/rollback", headers={"X-API-Key": "agent-key"}).status_code == 403
    r = client.post("/admin/kb/update", headers={"X-API-Key": "kbm-key"})
    assert r.status_code == 200 and r.json()["status"] == "no_changes"
    assert client.post("/admin/kb/rollback", headers={"X-API-Key": "admin-key"}).status_code == 409   # nothing to roll back to


def test_bad_attachment_base64_is_rejected(client):
    r = client.post("/chat", json={"customer_id": "c1", "message": "hi", "attachments": [{"filename": "a.png", "content_base64": "%%%"}]})
    assert r.status_code == 400


def test_unsafe_attachment_is_rejected_via_api(client):
    data = base64.b64encode(b"MZ\x90\x00").decode()
    body = client.post("/chat", json={"customer_id": "c1", "message": "see file",
                                      "attachments": [{"filename": "x.exe", "content_base64": data}]}).json()
    assert body["attachments"][0]["status"] == "rejected"


def test_input_validation(client):
    assert client.post("/chat", json={"customer_id": "", "message": "x"}).status_code == 422
    assert client.post("/chat", json={"customer_id": "c", "message": "x" * 5000}).status_code == 422
