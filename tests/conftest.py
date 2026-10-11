from datetime import datetime, timezone

import pytest

from a_real_time_gen_ai_customer_service_bot.core.clock import FakeClock


@pytest.fixture
def fake_clock():
    """A FakeClock starting Monday 12 Oct 2026, 10:00 UTC."""
    return FakeClock(datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc))
