from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from a_real_time_gen_ai_customer_service_bot.core.clock import FakeClock
from a_real_time_gen_ai_customer_service_bot.core.config import load_config
from a_real_time_gen_ai_customer_service_bot.core.business_calendar import BusinessCalendar

IST = ZoneInfo("Asia/Kolkata")


def ist(day, hour=0, minute=0, month=10, year=2026):
    return datetime(year, month, day, hour, minute, tzinfo=IST)


@pytest.fixture
def fake_clock():
    """A FakeClock starting Monday 12 Oct 2026, 10:00 UTC (15:30 IST)."""
    return FakeClock(datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc))


@pytest.fixture
def config():
    """A fresh Config loaded from config/default.yaml for each test."""
    return load_config()


@pytest.fixture
def clock():
    """Monday 12 Oct 2026, 11:00 IST: inside business hours."""
    return FakeClock(ist(12, 11, 0))


@pytest.fixture
def calendar(config):
    return BusinessCalendar(config)


from a_real_time_gen_ai_customer_service_bot.core.auth import AuthorizationService
from a_real_time_gen_ai_customer_service_bot.rag.answerer import KnowledgeAssistant
from a_real_time_gen_ai_customer_service_bot.rag.loader import load_directory
from a_real_time_gen_ai_customer_service_bot.rag.retriever import KnowledgeIndex, Retriever
from a_real_time_gen_ai_customer_service_bot.core.config import PROJECT_ROOT

KB_DIR = PROJECT_ROOT / "data" / "kb_docs"


@pytest.fixture
def auth(config):
    return AuthorizationService(config)


@pytest.fixture
def kb_docs():
    docs, errors = load_directory(KB_DIR)
    assert not errors
    return docs


@pytest.fixture
def assistant(config, clock, auth, kb_docs):
    index = KnowledgeIndex(kb_docs)
    return KnowledgeAssistant(Retriever(lambda: index, auth, config), config, clock)
