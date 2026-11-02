from datetime import datetime, timedelta, timezone

import pytest

from a_real_time_gen_ai_customer_service_bot.core.clock import Clock, FakeClock, SystemClock

START = datetime(2026, 10, 12, 10, 0, tzinfo=timezone.utc)


def test_system_clock_returns_timezone_aware_datetime():
    assert SystemClock().now().tzinfo is not None


def test_system_clock_and_fake_clock_are_clocks():
    assert isinstance(SystemClock(), Clock) and isinstance(FakeClock(START), Clock)


def test_fake_clock_starts_at_given_time(fake_clock):
    assert fake_clock.now() == START


def test_advance_moves_time_forward(fake_clock):
    fake_clock.advance(minutes=16)
    assert fake_clock.now() == START + timedelta(minutes=16)


def test_time_does_not_move_on_its_own(fake_clock):
    assert fake_clock.now() == fake_clock.now()


def test_set_can_jump_forward_and_backward(fake_clock):
    fake_clock.set(datetime(2026, 12, 25, 2, 0, tzinfo=timezone.utc))
    assert fake_clock.now().month == 12
    fake_clock.set(datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert fake_clock.now().month == 1


def test_multiple_advances_accumulate(fake_clock):
    fake_clock.advance(minutes=10)
    fake_clock.advance(minutes=5)
    fake_clock.advance(hours=1)
    assert fake_clock.now() == START + timedelta(hours=1, minutes=15)


def test_advance_zero_and_across_midnight(fake_clock):
    fake_clock.advance(minutes=0)
    assert fake_clock.now() == START
    fake_clock.advance(hours=14)
    assert fake_clock.now() == datetime(2026, 10, 13, 0, 0, tzinfo=timezone.utc)


def test_advance_negative_raises_and_keeps_time(fake_clock):
    with pytest.raises(ValueError):
        fake_clock.advance(minutes=-5)
    assert fake_clock.now() == START


def test_naive_datetimes_rejected(fake_clock):
    with pytest.raises(ValueError):
        FakeClock(datetime(2026, 10, 12, 10, 0))
    with pytest.raises(ValueError):
        fake_clock.set(datetime(2026, 10, 12, 10, 0))


def test_two_fake_clocks_are_independent():
    a, b = FakeClock(START), FakeClock(START)
    a.advance(hours=3)
    assert b.now() == START
